import os

from api.secret_manager import SecretManager
from api.storage import Storage
from dotenv import load_dotenv


if __name__ == '__main__':
    load_dotenv()
    access_key = os.environ.get('ACCESS_KEY')
    secret_key = os.environ.get('SECRET_KEY')

    storage = Storage(access_key=access_key, secret_key=secret_key)
    secret_manager = SecretManager(storage=storage)

    secret_manager.upload('test.dat', '../data/haloprop_halo0_snap84.dat')


    f = secret_manager.read('test.dat')
    print(f)