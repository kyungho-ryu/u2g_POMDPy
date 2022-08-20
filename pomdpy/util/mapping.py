import hashlib

def get_key(value):
    return hashlib.sha256(str(value).encode()).hexdigest()