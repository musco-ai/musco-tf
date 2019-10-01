class CompressionError(Exception):
    def __init__(self, *args):
        Exception.__init__(self, *args)
