from utils import ServerSetter


def main():
    worker = ServerSetter()
    worker.init_directories()
    worker.download_unpack_data()
    worker.prepare_data()


if __name__ == '__main__':
    main()
