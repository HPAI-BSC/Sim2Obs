import hvac


class VaultApi(object):
    def __init__(self, token):
        self.client = hvac.Client(url='https://pass.hpai.bsc.es:18200', token=token)

    def read_vault_secret(self, path=None, mount_point=None):
        return self.client.secrets.kv.v1.read_secret(path=path, mount_point=mount_point)

    def is_authenticated(self):
        return self.client.is_authenticated()


