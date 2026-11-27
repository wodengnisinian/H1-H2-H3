"""R36 remote helper: upload, run training/inference, run stats, poll, download."""
import argparse
import os
import posixpath
import stat

import paramiko

HOST = "tdklgrqctvo9g2techo.funhpc.com"
PORT = 21187
USER = "root"
PASSWORD = os.environ["SSH_PASSWORD"]

LOCAL = r"C:\Users\xxs\Documents\ChatGPT\speedweb\H1-H2-H3\experiments\R3-0036_冲击双门槛_20261127"
REMOTE = "/data/spiderweb-r36-20261127"
PY = "/data/miniconda/envs/torch/bin/python"
INPUTS = "/data/spiderweb-r3-20261124-1740"
COHORT_SRC = "/data/spiderweb-r23-20261127/data/cohort/cohort.jsonl.gz"
COHORT_SHA = "cf7d3ba220b789d978a56667409a3b6b5deb4832b8556d81d96f615acd05c1ad"


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
    for sub in ("", "/code", "/configs", "/results", "/results/prediction", "/results/stats",
                "/logs", "/data", "/data/cohort"):
        path = REMOTE + sub
        try:
            sftp.stat(path)
        except FileNotFoundError:
            sftp.mkdir(path)
    for name in ("code/r36_push.py", "code/r29_mix.py", "code/r27_entmax_combo.py",
                 "code/r26_sharp_cue.py", "code/r18_stats.py", "configs/protocol.json"):
        local = os.path.join(LOCAL, name)
        remote = posixpath.join(REMOTE, name)
        sftp.put(local, remote)
        print("uploaded", name)
    sftp.close()
    out, err, code = run(client,
        f"cp -f {COHORT_SRC} {REMOTE}/data/cohort/cohort.jsonl.gz && sha256sum {REMOTE}/data/cohort/cohort.jsonl.gz")
    print(out.strip())
    print("ERR", err.strip(), "rc", code)
    if COHORT_SHA not in out:
        raise SystemExit("cohort sha mismatch")


def launch(client, stage):
    common = (f"--b13 {INPUTS}/inputs/b13 --b16 {INPUTS}/inputs/b16 --prior {INPUTS}/inputs/prior "
              f"--cohort {REMOTE}/data/cohort/cohort.jsonl.gz")
    if stage == 1:
        argv = f"{PY} -B -u {REMOTE}/code/r36_push.py {common} --r29_code {REMOTE}/code --out {REMOTE}/results/prediction"
        log = f"{REMOTE}/logs/stage1.log"
    else:
        argv = (f"{PY} -B -u {REMOTE}/code/r18_stats.py --cohort {REMOTE}/data/cohort/cohort.jsonl.gz "
                f"--predictions {REMOTE}/results/prediction/predictions.jsonl.gz --primary lin3_lam12_role05 "
                f"--out {REMOTE}/results/stats")
        log = f"{REMOTE}/logs/stage2.log"
    cmd = f"cd {REMOTE} && nohup {argv} > {log} 2>&1 & echo LAUNCHED $!"
    out, err, code = run(client, cmd)
    print(out.strip(), err.strip(), "rc", code)


def status(client):
    out, err, code = run(client,
        f"ps -eo pid,etime,cmd | grep -E 'r36_(push|stats)' | grep -v grep; "
        f"echo '--- stage1 tail ---'; tail -n 12 {REMOTE}/logs/stage1.log 2>/dev/null; "
        f"echo '--- stage2 tail ---'; tail -n 20 {REMOTE}/logs/stage2.log 2>/dev/null; "
        f"echo '--- results ---'; ls -la {REMOTE}/results/prediction {REMOTE}/results/stats 2>/dev/null")
    print(out)
    print("ERR", err.strip())


def download(client):
    sftp = client.open_sftp()
    for remote_sub, local_sub in (("/results/prediction", "/results/prediction"),
                                  ("/results/stats", "/results/stats"),
                                  ("/logs", "/logs")):
        for entry in sorted(sftp.listdir_attr(REMOTE + remote_sub), key=lambda e: e.filename):
            name = entry.filename
            if stat.S_ISDIR(entry.st_mode):
                continue
            remote = posixpath.join(REMOTE + remote_sub, name)
            local = os.path.join(LOCAL + local_sub.replace("/", os.sep), name)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            sftp.get(remote, local)
            print("downloaded", local_sub + "/" + name)
    sftp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("upload", "run1", "run2", "status", "download"))
    args = parser.parse_args()
    client = connect()
    try:
        if args.action == "upload":
            upload(client)
        elif args.action in ("run1", "run2"):
            launch(client, int(args.action[-1]))
        elif args.action == "status":
            status(client)
        elif args.action == "download":
            download(client)
    finally:
        client.close()


if __name__ == "__main__":
    main()
