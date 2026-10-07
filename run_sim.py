# run_sim.py
# P2P Federated + Continual Learning + Fault Tolerance + Graceful Ctrl+C Shutdown

import time
import socket
import pickle
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.cnn_model import CNNModel
from utils.data_loader import ReplayBuffer, SkinCancerDataset, ReplaySkinCancerDataset
from utils.dynamic_data import ingest_new_data, partition_new_samples
import threading
import logging
import psutil
import random
import signal
import sys
from sklearn.metrics import precision_recall_fscore_support

# --------------------------------------------------------------
# Logging
# --------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("P2P-FL")

# --------------------------------------------------------------
# Hyper-parameters
# --------------------------------------------------------------
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE       = 8
LR               = 0.0001
NUM_ROUNDS       = 10
INITIAL_CLIENTS  = 3
PROXIMAL_MU      = 0.1
EXCHANGE_TIMEOUT = 8.0
MAX_CLIENTS      = 5

BUFFER_SIZE      = 200
REPLAY_RATIO     = 0.5

INFINITE_MODE    = True
IDLE_SLEEP_SEC   = 60
IDLE_THRESHOLD   = 5

CRASH_CLIENT_ID     = 2
CRASH_AFTER_ROUND   = 3

ENABLE_NET_FAILURES = True
FAILURE_PROB_DROP   = 0.10
LINK_FAILURE_START  = 5
LINK_FAILURE_END    = 7
NET_FAIL_THRESHOLD  = 3
RECOVERY_ROUND      = 9

print(f"Using device: {DEVICE}")
print(f"Infinite Mode: {INFINITE_MODE} | Ctrl+C to stop anytime with final results")

# ==============================================================
# PORT CLEANUP
# ==============================================================
def free_port_if_held(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port and conn.status in ('LISTEN', 'TIME_WAIT'):
            try:
                psutil.Process(conn.pid).terminate()
                logger.warning(f"[INIT] Killed stale process on port {port}")
            except:
                pass

def is_port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True
        except:
            return False

active_cids = []
for cid in range(INITIAL_CLIENTS):
    port = 8000 + cid
    free_port_if_held(port)
    if is_port_free(port):
        active_cids.append(cid)

if not active_cids:
    raise RuntimeError("No clients can start! Ports 8000-8002 busy.")

logger.info(f"Starting clients: {active_cids}")

# ==============================================================
# FAILURE SIMULATOR
# ==============================================================
def simulate_failure(cid: int, round_num: int, action: str = 'send') -> bool:
    if not ENABLE_NET_FAILURES: return False
    if random.random() < FAILURE_PROB_DROP:
        logger.warning(f"[NETFAIL] Random {action} drop C{cid}")
        return True
    if cid == 1 and LINK_FAILURE_START <= round_num <= LINK_FAILURE_END:
        logger.warning(f"[NETFAIL] Link outage C{cid}")
        return True
    if action == 'bind' and random.random() < 0.3:
        time.sleep(random.uniform(1.0, 3.0))
    return False

# ==============================================================
# CLIENT CLASS
# ==============================================================
class LocalClient:
    def __init__(self, cid):
        self.cid = cid
        self.model = CNNModel(pretrained=True, freeze_backbone=False).to(DEVICE)
        self.data_dir = f"clients_data/{cid}"
        self.train_ds = SkinCancerDataset(self.data_dir, "train")
        self.test_ds  = SkinCancerDataset(self.data_dir, "test")

        self.replay_buffer = ReplayBuffer(BUFFER_SIZE)
        init_samples = random.sample(self.train_ds.data, min(BUFFER_SIZE, len(self.train_ds.data)))
        self.replay_buffer.add_samples(init_samples)

        self.reload_datasets()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR, weight_decay=1e-4)
        self.consecutive_failures = 0

    def reload_datasets(self, new_samples=None):
        if new_samples:
            self.train_ds.reload(new_samples, self.replay_buffer)
        self.train_ds = ReplaySkinCancerDataset(
            self.data_dir, "train", transform=self.train_ds.transform,
            replay_buffer=self.replay_buffer, replay_ratio=REPLAY_RATIO
        )
        self.train_loader = DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        self.test_loader  = DataLoader(self.test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    def get_params(self): return [p.detach().cpu().numpy().astype(np.float32) for p in self.model.parameters()]
    def set_params(self, params):
        sd = self.model.state_dict()
        for i, (n, _) in enumerate(self.model.named_parameters()):
            sd[n] = torch.tensor(params[i]).to(DEVICE)
        self.model.load_state_dict(sd)
    def average_params(self, p1, p2): return [(a + b) / 2.0 for a, b in zip(p1, p2)]

    def train_epoch(self, global_params=None):
        if global_params is None: global_params = self.get_params()
        self.set_params(global_params)
        g_tensors = [torch.tensor(p).to(DEVICE) for p in global_params]

        self.model.train()
        loss_sum = 0.0
        for x, y in self.train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.criterion(out, y)
            if PROXIMAL_MU > 0:
                prox = sum(((lp - gp)**2).sum() for lp, gp in zip(self.model.parameters(), g_tensors))
                loss += (PROXIMAL_MU / 2) * prox
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            loss_sum += loss.item()
        logger.info(f"[C{self.cid}] Train loss: {loss_sum/len(self.train_loader):.4f}")
        return self.get_params()

    def evaluate(self):
        self.model.eval()
        preds, labels_list = [], []
        loss_sum = total = 0.0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = self.model(x)
                loss_sum += self.criterion(out, y).item() * y.size(0)
                pred = out.argmax(1)
                preds.extend(pred.cpu().numpy())
                labels_list.extend(y.cpu().numpy())
                total += y.size(0)

        acc = np.mean(np.array(preds) == np.array(labels_list))
        avg_loss = loss_sum / total if total > 0 else 0.0

        if len(set(labels_list)) > 1:
            p, r, f1, _ = precision_recall_fscore_support(labels_list, preds, average='macro', zero_division=0)
        else:
            p = r = f1 = 0.0

        

        return avg_loss, acc, p, r, f1

# ==============================================================
# P2P COMMUNICATION
# ==============================================================
def robust_bind_and_listen(cid, round_num):
    for _ in range(3):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(('localhost', 8000 + cid))
            simulate_failure(cid, round_num, 'bind')
            s.listen(1)
            return s
        except:
            time.sleep(1)
    return None

def robust_send(cid, dst_cid, params, round_num):
    if simulate_failure(cid, round_num, 'send'): return False
    try:
        with socket.socket() as s:
            s.settimeout(EXCHANGE_TIMEOUT)
            s.connect(('localhost', 8000 + dst_cid))
            s.sendall(pickle.dumps(params))
        return True
    except:
        return False

def robust_accept_one(cid, sock, round_num):
    if simulate_failure(cid, round_num, 'recv'): return False, None
    try:
        sock.settimeout(EXCHANGE_TIMEOUT)
        conn, _ = sock.accept()
        data = b''.join(iter(lambda: conn.recv(1024*1024), b''))
        conn.close()
        sock.close()
        return True, pickle.loads(data)
    except:
        return False, None

def p2p_aggregate(clients, round_num):
    if len(clients) <= 1: return

    socks = {c.cid: robust_bind_and_listen(c.cid, round_num) for c in clients}
    time.sleep(1.2)

    for i, c in enumerate(clients):
        robust_send(c.cid, clients[(i+1)%len(clients)].cid, c.get_params(), round_num)

    results = {}
    threads = []
    for i, c in enumerate(clients):
        sock = socks.get(c.cid)
        if not sock:
            results[c.cid] = (False, None)
            continue
        t = threading.Thread(target=lambda: results.update(
            {c.cid: robust_accept_one(c.cid, sock, round_num)}), daemon=True)
        t.start()
        threads.append(t)
    for t in threads: t.join(EXCHANGE_TIMEOUT + 3)

    for i, c in enumerate(clients):
        ok, params = results.get(c.cid, (False, None))
        if ok and params is not None:
            params = c.average_params(c.get_params(), params)
            c.set_params(params)
            c.consecutive_failures = max(0, c.consecutive_failures - 1)
        else:
            c.consecutive_failures += 1

    to_remove = [c for c in clients if c.consecutive_failures >= NET_FAIL_THRESHOLD]
    if to_remove:
        logger.critical(f"[ADAPT] Removing clients {[c.cid for c in to_remove]}")
        clients[:] = [c for c in clients if c not in to_remove]

# ==============================================================
# GRACEFUL SHUTDOWN SETUP
# ==============================================================
shutdown_requested = False

def signal_handler(sig, frame):
    global shutdown_requested
    if shutdown_requested:
        logger.critical("Forced exit!")
        sys.exit(0)
    shutdown_requested = True
    logger.critical("\n" + "="*70)
    logger.critical("Ctrl+C detected – stopping after current round...")
    logger.critical("Final evaluation will be shown shortly.")
    logger.critical("="*70)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ==============================================================
# MAIN LOOP
# ==============================================================
try:
    clients = [LocalClient(cid) for cid in active_cids]
    prev_round_models = [c.get_params() for c in clients]
    round_num = 1
    idle_rounds = 0

    while (INFINITE_MODE or round_num <= NUM_ROUNDS) and not shutdown_requested:
        logger.info(f"\n{'='*25} ROUND {round_num} {'='*25}")

        # Recovery
        if len(clients) < INITIAL_CLIENTS // 2 or round_num == RECOVERY_ROUND:
            for cid in set(range(INITIAL_CLIENTS)) - {c.cid for c in clients}:
                if is_port_free(8000 + cid):
                    new_c = LocalClient(cid)
                    if clients:
                        avg_p = clients[0].get_params()
                        for c in clients[1:]:
                            avg_p = new_c.average_params(avg_p, c.get_params())
                        new_c.set_params(avg_p)
                    clients.append(new_c)
                    prev_round_models.append(new_c.get_params())
                    logger.critical(f"[RECOVER] Re-added Client {cid}")

        # Crash simulation
        if CRASH_AFTER_ROUND > 0 and round_num == CRASH_AFTER_ROUND and any(c.cid == CRASH_CLIENT_ID for c in clients):
            logger.critical(f"[SIM] CRASHING Client {CRASH_CLIENT_ID}")
            clients = [c for c in clients if c.cid != CRASH_CLIENT_ID]
            prev_round_models = [c.get_params() for c in clients]

        if not clients:
            logger.critical("All clients dead!")
            break

        # Local training
        for i, c in enumerate(clients):
            proxy = prev_round_models[(i - 1) % len(clients)]
            prev_round_models[i] = c.train_epoch(proxy)

        # P2P aggregation
        p2p_aggregate(clients, round_num)

        # Evaluation
        if clients:
            losses, accs, ps, rs, fs = [], [], [], [], []
            for c in clients:
                loss, acc, p, r, f1 = c.evaluate()
                losses.append(loss); accs.append(acc); ps.append(p); rs.append(r); fs.append(f1)
                logger.info(f"[R{round_num}] C{c.cid} → Loss:{loss:.4f} Acc:{acc:.4f} P:{p:.4f} R:{r:.4f} F1:{f1:.4f}")
            logger.info(f"[R{round_num}] AVG → Loss:{np.mean(losses):.4f} Acc:{np.mean(accs):.4f} "
                        f"P:{np.nanmean(ps):.4f} R:{np.nanmean(rs):.4f} F1:{np.nanmean(fs):.4f}")

        # Dynamic data
        new_samples = ingest_new_data()
        if new_samples:
            idle_rounds = 0
            partitions = partition_new_samples(new_samples, len(clients))
            for c, part in zip(clients, partitions):
                c.reload_datasets(part)
        else:
            idle_rounds += 1
            if INFINITE_MODE and idle_rounds % IDLE_THRESHOLD == 0:
                logger.info(f"[IDLE] No new data for {idle_rounds} rounds. Sleeping {IDLE_SLEEP_SEC}s...")
            if INFINITE_MODE:
                time.sleep(IDLE_SLEEP_SEC)

        prev_round_models = [c.get_params() for c in clients]
        round_num += 1

except KeyboardInterrupt:
    pass
finally:
    # FINAL EVALUATION ON EXIT
    logger.info("\n" + "="*80)
    logger.info("GRACEFUL SHUTDOWN – FINAL RESULTS")
    logger.info("="*80)

    if clients:
        final_l = final_a = final_p = final_r = final_f1 = []
        for c in clients:
            loss, acc, p, r, f1 = c.evaluate()
            final_l.append(loss); final_a.append(acc)
            final_p.append(p); final_r.append(r); final_f1.append(f1)
            logger.info(f"FINAL | Client {c.cid} → Loss:{loss:.4f} Acc:{acc:.4f} "
                        f"Precision:{p:.4f} Recall:{r:.4f} F1:{f1:.4f}")

        logger.info("-" * 80)
        logger.info(f"FINAL AVERAGE → "
                    f"Loss: {np.mean(final_l):.4f} | "
                    f"Accuracy: {np.mean(final_a):.4f} | "
                    f"Precision: {np.nanmean(final_p):.4f} | "
                    f"Recall: {np.nanmean(final_r):.4f} | "
                    f"F1-Score: {np.nanmean(final_f1):.4f}")
        logger.info(f"Simulation stopped at Round {round_num-1} | Active clients: {[c.cid for c in clients]}")
    else:
        logger.warning("No clients survived.")

    logger.info("="*80)
    logger.info("Simulation terminated gracefully. Thank you!")
    logger.info("="*80)