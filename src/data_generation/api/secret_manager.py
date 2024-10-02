from api.storage import Storage


class SecretManager:
    def __init__(self, storage: Storage):
        self.storage = storage

    def read(self, file_name):
        df = None
        if file_name.endswith('csv'):
            df = self.storage.load_minio_client().load_csv_pd(file_name)
        elif file_name.endswith('.dat'):
            df = self.storage.load_minio_client().load_text(file_name)
        return df

    def upload(self, final_path, file_name):
        self.storage.load_minio_client().upload_file(final_path, file_name)
