class CentralContainer:
    central_container = {}

    @staticmethod
    def get(key):
        return CentralContainer.central_container.get(key)

    @staticmethod
    def get_and_delete(key):
        value = CentralContainer.get(key)
        if key in CentralContainer.central_container:
            del CentralContainer.central_container[key]
        return value

    @staticmethod
    def put(key, value):
        CentralContainer.central_container[key] = value
