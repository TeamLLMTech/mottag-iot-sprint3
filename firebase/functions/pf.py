import numpy # Used for matrix operations in localization algorithm
from sys import version_info # Used to check the Python-interpreter version at runtime

# RSSI_Localizer
    # Use:
        # from rssi import RSSI_Localizer
        # rssi_localizer_instance = RSSI_Localizer()
    # -------------------------------------------------------
    # Description:
        # This class helps a user implement rssi-based localization.
        # The algorithm assumes the logarithmic distance-path-loss model
        # And assumes a minimum of 3 (or more) access points.
    # -------------------------------------------------------
    # Input:
        # accessPoints: Array holding accessPoint dictionaries.
        #               The order of the arrays supplied will retain
        #               its order, throughout the entire execution.
        # [{
        #     'signalAttenuation': 3, 
        #     'location': {
        #         'y': 1, 
        #         'x': 1
        #     }, 
        #     'reference': {
        #         'distance': 4, 
        #         'signal': -50
        #     }, 
        #     'name': 'dd-wrt'
        # },
        # {
        #     'signalAttenuation': 4, 
        #     'location': {
        #         'y': 1, 
        #         'x': 7
        #     }, 
        #     'reference': {
        #         'distance': 3, 
        #         'signal': -41
        #     }, 
        #     'name': 'ucrwpa'
        # }]
class RSSI_Localizer(object):
    # Allows us to fetch for networks/accessPoints externally.
    # Array of access points must be formatted.
    # 'self.count' parameter is computed internally to aid in 
    # scaling of the algorithm.
    def __init__(self,accessPoints):
        self.accessPoints = accessPoints
        self.count = len(accessPoints)

    # getDistanceFromAP
        # Description:
            # Uses the log model to compute an estimated dstance(di) from node(i)
        # -------------------------------------------------------
        # Input: 
            # accessPoint: dicitonary holding accesspoint info.
            # {
            #     'signalAttenuation': 3, 
            #     'location': {
            #         'y': 1, 
            #         'x': 1
            #     }, 
            #     'reference': {
            #         'distance': 4, 
            #         'signal': -50
            #     }, 
            #     'name': 'dd-wrt'
            # }
            # signalStrength: -69
        # -------------------------------------------------------
        # output: 
            # accessPoint: dicitonary holding accesspoint info.
            # {
            #     'signalAttenuation': 3, 
            #     'location': {
            #         'y': 1, 
            #         'x': 1
            #     }, 
            #     'reference': {
            #         'distance': 4, 
            #         'signal': -50
            #     }, 
            #     'name': 'dd-wrt',
            #     'distance': 2
            # }
    @staticmethod
    def getDistanceFromAP(accessPoint, signalStrength):
        beta_numerator = float(accessPoint['reference']['signal']-signalStrength)
        beta_denominator = float(10*accessPoint['signalAttenuation'])
        beta = beta_numerator/beta_denominator
        distanceFromAP = round(((10**beta)*accessPoint['reference']['distance']),4)
        accessPoint.update({'distance':distanceFromAP})
        return accessPoint
    
    # TODO fix this because theres two consecutive for loops. 
    # One that runs to fefd signal strengths to this function, 
    # a second consecutive loop inside the function.

    # getDistancesForAllAPs
        # Description:
            # Makes use of 'getDistanceFromAP' to iterate through all 
            # accesspoints being used in localization and obtains the 
            # distance from each one of them.
        # ------------------------------------------------
        # Input:
            # signalStrengths:
            # [siganl1, siganl2, siganl3]
            # [-42, -53, -77]
        # ------------------------------------------------
        # Output:
            # [
            #     {
            #         'distance': 4,
            #         'x': 2,
            #         'y': 3
            #     },
            #     {
            #         'distance': 7,
            #         'x': 2,
            #         'y': 5
            #     },
            #     {
            #         'distance': 9,
            #         'x': 7,
            #         'y': 3
            #     }
            # ]
    def getDistancesForAllAPs(self, signalStrengths):
        apNodes = []
        for i in range(len(self.accessPoints)):
            ap = self.accessPoints[i] 
            distanceFromAP = self.getDistanceFromAP(
                ap,
                signalStrengths[i]
            )
            apNodes.append({
                'distance': distanceFromAP['distance'],
                'x': ap['location']['x'],
                'y': ap['location']['y']
            })
        return apNodes
    
    # createMatrices
        # Description:
            # Creates tehmatrices neccesary to use the least squares method
            # in order to mnimize the error (error=|realDistance-estimatedDistance|). 
            # Assuming 'n' number of nodes and d(m) is the distance(d) from node (m).
            # AX = B, where X is our estimated location.
            # A = [
            #     2(x(i)-xn)    2(y(i)-yn)
            #     2(x(i+1)-xn)  2(y(i+1)-yn)
            #     ...           ...
            #     2(x(n-1)-xn)  2(y(n-1)-yn)
            # ]
            # B = [
            #     x(i)^2 + y(i)^2 - x(n)^2 + y(n)^2 - d(i)^2 + d(n)^2
            #     x(i+1)^2 + y(i+1)^2 - x(n)^2 + y(n)^2 - d(i+1)^2 + d(n)^2
            #     ...
            #     x(n-1)^2 + y(n-1)^2 - x(n)^2 + y(n)^2 - d(n-1)^2 + d(n)^2
            # ]
        # ----------------------------------------
        # Input:
            # accessPoints
            # [
            #     {
            #         'distance': 4,
            #         'x': 2,
            #         'y': 3
            #     },
            #     {
            #         'distance': 7,
            #         'x': 2,
            #         'y': 5
            #     },
            #     {
            #         'distance': 9,
            #         'x': 7,
            #         'y': 3
            #     }
            # ]
        # ----------------------------------------
        # Output:
            # A = [
            #     2(2-7)    2(3-3)
            #     2(2-7)  2(5-3)
            # ]
            # B = [
            #     2^2 + 3^2 - 7^2 + 3^2 - 4^2 + 9^2
            #     2^2 + 5^2 - 7^2 + 3^2 - 7^2 + 9^2
            # ]
    def createMatrices(self, accessPoints):
        # Sets up that te matrics only go as far as 'n-1' rows,
        # with 'n being the # of access points being used.
        n_count = self.count-1
        # initialize 'A' matrix with 'n-1' ranodm rows.
        a = numpy.empty((n_count,2))
        # initialize 'B' matrix with 'n-1' ranodm rows.
        b = numpy.empty((n_count,1))
        # Define 'x(n)' (x of last accesspoint)
        x_n = accessPoints[n_count]['x'] 
        # Define 'y(n)' (y of last accesspoint)
        y_n = accessPoints[n_count]['y']
        # Define 'd(n)' (distance from of last accesspoint)
        d_n = accessPoints[n_count]['distance']
        # Iteration through accesspoints is done upto 'n-1' only
        for i in range(n_count):
            ap = accessPoints[i]
            x, y, d = ap['x'], ap['y'], ap['distance']
            a[i] = [2*(x-x_n), 2*(y-y_n)]
            b[i] = [(x**2)+(y**2)-(x_n**2)-(y_n**2)-(d**2)+(d_n**2)]
        return a, b
    
    # computePosition
        # Description:
            # Performs the 'least squares method' matrix operations 
            # neccessary to get the 'x' and 'y' of the unknown 
            # beacon's position.
            # X = [(A_transposed*A)^-1]*[A_transposed*B]
        # ----------------------------------------
        # Input:
            # A = [
            #     0   0
            #     0  -4
            # ]
            # B = [
            #     4 + 9 - 49 + 9 - 16 + 81  => 38
            #     4 + 25 - 49 + 9 - 49 + 81 => 21
            # ]
        # ----------------------------------------
        # Output:
            # x
            # [
            #     2,
            #     3
            # ]
    @staticmethod
    def computePosition(a, b):
        # Get 'A_transposed' matrix
        at = numpy.transpose(a)
        # Get 'A_transposed*A' matrix
        at_a = numpy.matmul(at,a)
        # Get '[(A_transposed*A)^-1]' matrix
        inv_at_a = numpy.linalg.inv(at_a)
        # Get '[A_transposed*B]'
        at_b = numpy.matmul(at,b)
        # Get '[(A_transposed*A)^-1]*[A_transposed*B]'
        # This holds our position (xn,yn)
        x = numpy.matmul(inv_at_a,at_b) 
        return x

    # getNodePosition
        # Description:
            # Combines 'getDistancesForAllAPs', 'createMatrics',
            # and 'computerPosition' to get the 'X' vector that
            # contains our unkown (x,y) position.
        # ----------------------------------------
        # Input:
            # signalStrengths
            # [4, 2 , 3]
        # ----------------------------------------
        # Output:
            # x
            # [2, 3]
    def getNodePosition(self, signalStrengths):
        apNodes = self.getDistancesForAllAPs(signalStrengths)
        a, b = self.createMatrices(apNodes) 
        position = self.computePosition(a, b)
        # print(a)
        # print(b)
        return position