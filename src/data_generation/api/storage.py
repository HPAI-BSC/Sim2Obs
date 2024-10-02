from hpai_utils.minio import Minio


class Storage(object):
    def __init__(self, access_key, secret_key):
        self.access_key = access_key
        self.secret_key = secret_key

    def load_minio_client(self, strongbox_path='strongbox/space/read-write') -> Minio(None, None):
        try:
            minio_client = Minio('https://storage.hpai.bsc.es:8080/',
                                 'space',
                                 access_key=self.access_key,
                                 secret_key=self.secret_key)
            return minio_client
        except Exception as e:
            print('Cannot load minio client with error: {}'.format(e))
            raise e
