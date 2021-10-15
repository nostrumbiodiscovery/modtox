class FeatureError(Exception):
    pass

class ClusterError(Exception):
    pass

class ServerError(Exception):
    def __init__(self, db):
        message = f"Impossible to access {db} server. Try again later."
        super().__init__(message)
        
class BadRequestError(Exception):
    def __init__(self, db, uniprot):
        message = f"Target {uniprot} not present in {db} database."
        super().__init__(message)

class UnsupportedStandardRelation(Exception):
    pass

class UnsupportedStandardType(Exception):
    pass

class BalancingError(Exception):
    pass

class ScalingError(Exception):
    pass

