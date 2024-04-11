"""
Created By: ishwor subedi
Date: 2024-03-29
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.api.fastapi:app", host="0.0.0.0", port=8000)
