import os
import sys
import threading
import time

import uvicorn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dry-run', action='store_true', help='dry run')
    args = parser.parse_args()
    config = uvicorn.Config("rest.rest:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=[os.path.join(os.getcwd(), "src")])
    server = uvicorn.Server(config)
    if args.dry_run:
        threading.Thread(target=server.run).start()
        while True:
            if server.started:
                server.shutdown()
                os.kill(os.getpid(), 9)
            time.sleep(10)

    server.run()
