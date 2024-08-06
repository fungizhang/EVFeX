
import grpc

def getGrpcCredentials(crtPath):
    with open(crtPath, 'rb') as f:
        trusted_certs = f.read()
        return grpc.ssl_channel_credentials(root_certificates=trusted_certs)