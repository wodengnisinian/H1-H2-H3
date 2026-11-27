"""R43 remote helper: upload frozen candidate, run one-shot H3 acceptance, download."""
import argparse
import os
import posixpath
import stat

import paramiko

HOST = "tdklgrqctvo9g2techo.funhpc.com"
PORT = 21187
USER = "root"
PASSWORD = os.environ["SSH_PASSWORD"]

LOCAL = r"C:\Users\xxs\Documents\ChatGPT\speedweb\H1-H2-H3\experiments\R3-0044_最终算法复测_20261127"
R21_FEATURES = r"C:\Users\xxs\Documents\ChatGPT\speedweb\H1-H2-H3\experiments\R3-0021_位置分数H1H2H3_20261126\results\h3\features.jsonl"
REMOTE = "/data/spiderweb-r44-20261127"
PY = "/data/miniconda/envs/torch/bin/python"
INPUTS = "/data/spiderweb-r3-20261124-1740"
TEMPLATES = INPUTS + "/R3/configs/formal_templates.json"
R16_COHORT = "/data/spiderweb-r23-20261127/data/cohort/cohort.jsonl.gz"
R09_FEATURES = INPUTS + "/R3/experiments/11-R09/results/features.jsonl"
R24_FEATURES = "/data/spiderweb-r24-20261127/results/h3/features.jsonl"
R42_FEATURES = "/data/spiderweb-r42-20261127/results/h3/features.jsonl"
R43_FEATURES = "/data/spiderweb-r43-20261127/results/h3/features.jsonl"


def connect():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, port=PORT, username=USER, password=PASSWORD, timeout=30)
    return client


def run(client, cmd, timeout=7200):
    stdin, stdout, stderr = client.exec_command(cmd, timeout=7200)
    out = stdout.read().decode("utf8", "replace")
    err = stderr.read().decode("utf8", "replace")
    code = stdout.channel.recv_exit_status()
    return out, err, code


def upload(client):
    sftp = client.open_sftp()
    for sub in ("", "/code", "/configs", "/checkpoints", "/refs", "/logs"):
        path = REMOTE + sub
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)
    for name in ("code/r44_h3.py", "code/r44_evaluate.py", "code/r44_module.py",
                 "configs/protocol.json"):
        sftp.put(os.path.join(LOCAL, name), posixpath.join(REMOTE, name))
        print("uploaded", name)
    for seed in (1701, 2701, 3701):
        name = f"checkpoints/mix2raug-{seed}.json"
        sftp.put(os.path.join(LOCAL, name), posixpath.join(REMOTE, name))
        print("uploaded", name)
    sftp.put(R21_FEATURES, REMOTE + "/refs/r21_features.jsonl")
    print("uploaded refs/r21_features.jsonl")
    sftp.close()


def launch(client):
    argv = (f"{PY} -B -u {REMOTE}/code/r44_h3.py all "
            f"--b13 {INPUTS}/inputs/b13 --b16 {INPUTS}/inputs/b16 --prior {INPUTS}/inputs/prior "
            f"--checkpoints {REMOTE}/checkpoints --templates {TEMPLATES} "
            f"--r16_cohort {R16_COHORT} --r09_features {R09_FEATURES} "
            f"--r21_features {REMOTE}/refs/r21_features.jsonl --r24_features {R24_FEATURES} "
            f"--r42_features {R42_FEATURES} --r43_features {R43_FEATURES} --out {REMOTE}/results/h3")
    cmd = f"cd {REMOTE} && nohup {argv} > logs/acceptance.log 2>&1 & echo LAUNCHED $!"
    out, err, code = run(client, cmd)
    print(out.strip(), err.strip(), "rc", code)


def status(client):
    out, err, code = run(client,
        f"ps -eo pid,etime,cmd | grep r43 | grep -v grep; "
        f"echo '--- log tail ---'; tail -n 15 {REMOTE}/logs/acceptance.log 2>/dev/null; "
        f"echo '--- results ---'; ls -la {REMOTE}/results/h3 2>/dev/null")
    print(out)
    print("ERR", err.strip())


def download(client):
    sftp = client.open_sftp()
    local_h3 = os.path.join(LOCAL, "results", "h3")
    os.makedirs(local_h3, exist_ok=True)
    try:
        entries = sftp.listdir_attr(REMOTE + "/results/h3")
    except FileNotFoundError:
        print("no h3 results yet")
        sftp.close()
        return
    for entry in sorted(entries, key=lambda e: e.filename):
        if not stat.S_ISREG(entry.st_mode):
            continue
        sftp.get(REMOTE + "/results/h3/" + entry.filename,
                 os.path.join(local_h3, entry.filename))
        print("downloaded results/h3/" + entry.filename)
    try:
        sftp.stat(REMOTE + "/logs/acceptance.log")
        sftp.get(REMOTE + "/logs/acceptance.log", os.path.join(LOCAL, "logs", "acceptance.log"))
        print("downloaded logs/acceptance.log")
    except FileNotFoundError:
        pass
    sftp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "run", "status", "download"))
    args = parser.parse_args()
    client = connect()
    try:
        if args.action == "upload":
            upload(client)
        elif args.action == "run":
            launch(client)
        elif args.action == "status":
            status(client)
        elif args.action == "download":
            download(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
