import os
import sys
import uvicorn


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == "__main__":
    uvicorn.run("rest.rest:app", host="0.0.0.0", port=8000, reload=True)
