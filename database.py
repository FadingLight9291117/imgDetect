import sqlite3
from pathlib import Path


def createDB(name='example.db'):
    if Path(name).exists():
        return
    con = sqlite3.connect(name)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE result (id INT, time float)
    """)
    con.commit()
    con.close()


if __name__ == '__main__':
    createDB()
