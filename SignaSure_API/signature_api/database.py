import hashlib
import sqlite3
import random

# Generates a random pin for the user
def generate_pin():
    pin = str(random.randint(100000,999999))
    return pin

# Hash the PIN using SHA-256
def hash_pin(pin):
    return hashlib.sha256(pin.encode()).hexdigest()

# Creates the SQLite database "pins.db"
def create_db():
    connection = sqlite3.connect("pins.db")
    cursor = connection.cursor()

    # Create table for storing PINs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pin_hash TEXT NOT NULL
        )
    ''')

    connection.commit()
    connection.close()


# Writes the PIN into the database
def store_pin(pin):
    connection = sqlite3.connect("pins.db")
    cursor = connection.cursor()

    # Converting the pin before storing it
    hashed_pin = hash_pin(pin)

    # Insert the PIN into the database
    cursor.execute("INSERT INTO pins (pin_hash) VALUES (?)", (hashed_pin,))
    
    connection.commit()
    connection.close()

# Check if the provided PIN exists in the database
def validate_pin(user_pin):
    conn = sqlite3.connect('pins.db')
    c = conn.cursor()
    user_pin_hash = hash_pin(user_pin)

    # Compare the hashed PIN with the stored hash
    c.execute('SELECT * FROM pins WHERE pin_hash = ?', (user_pin_hash,))
    result = c.fetchone()
    conn.close()
    
    return result is not None