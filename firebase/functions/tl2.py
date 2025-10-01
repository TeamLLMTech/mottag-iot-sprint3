

class TrilaterationController2:
    def __init__(
        self,
        bp_1: tuple,
        bp_2: tuple,
        bp_3: tuple,
        measured_power=-69,
        path_loss_exponent=1.8,
    ):
        """
        Initialize the trilateration controller.

        Args:
            bp_1 (tuple): Position of base station 1.
            bp_2 (tuple): Position of base station 2.
            bp_3 (tuple): Position of base station 3.
            measured_power (int, optional): Measured power at base station. Defaults to -69.
            path_loss_exponent (float, optional): Path loss exponent. Defaults to 1.8.
        """

        from scipy.optimize import least_squares
        self.least_squares = least_squares
        # Base station positions
        self.bp_1 = bp_1
        self.bp_2 = bp_2
        self.bp_3 = bp_3

        # Measured power and path loss exponent
        self.measured_power = measured_power
        self.path_loss_exponent = path_loss_exponent

    def get_position(self, rssi_1: float, rssi_2: float, rssi_3: float) -> tuple:
        """
        Calculates the estimated position based on the received signal strength indicator (RSSI) values
        and the known positions of three base stations.

        Args:
            rssi_1 (float): The RSSI value received from base station 1.
            rssi_2 (float): The RSSI value received from base station 2.
            rssi_3 (float): The RSSI value received from base station 3.

        Returns:
            tuple: The estimated position (x, y).
        """
        # Calculate distances
        d1 = self.get_distance(rssi_1)
        d2 = self.get_distance(rssi_2)
        d3 = self.get_distance(rssi_3)

        # Trilateration
        estimated_x, estimated_y = self.trilaterate(d1, d2, d3)

        return estimated_x, estimated_y

    def trilaterate(self, d1: float, d2: float, d3: float) -> tuple:
        """
        Trilaterates the position (X, Y) given the distances of three points.

        Args:
            d1 (float): distance from the first point to the unknown position.
            d2 (float): distance from the second point to the unknown position.
            d3 (float): distance from the third point to the unknown position.

        Returns:
            tuple: The (X, Y) coordinates of the unknown position.
        """
        x1, y1 = self.bp_1
        x2, y2 = self.bp_2
        x3, y3 = self.bp_3

        # Formula:
        # (x - x1)^2 + (y - y1)^2 = d1^2
        # (x - x2)^2 + (y - y2)^2 = d2^2
        # (x - x3)^2 + (y - y3)^2 = d3^2
        def equations(guess):
            x, y, r = guess

            return (
                (x - x1) ** 2 + (y - y1) ** 2 - (d1 - r) ** 2,
                (x - x2) ** 2 + (y - y2) ** 2 - (d2 - r) ** 2,
                (x - x3) ** 2 + (y - y3) ** 2 - (d3 - r) ** 2,
            )

        # Initial guess
        initial_guess = (0, 0, 0)

        # Use least squares to solve the equations
        results = self.least_squares(equations, initial_guess)

        # Return the estimated coordinates
        coordinates = results.x
        return coordinates[0], coordinates[1]

    def get_distance(self, rssi: float) -> float:
        """
        Converts RSSI (Received Signal Strength Indicator) to distance using the path loss model.

        Parameters:
        - rssi (float): The received signal strength indicator in dBm.

        Returns:
        - distance (float): The calculated distance between the devices in meters.
        """

        return 10 ** ((self.measured_power - rssi) / (10 * self.path_loss_exponent))

    # Scaling removed; coordinates are now returned as-is.

    def __str__(self):
        return f"TrilaterationController(bp_1={self.bp_1}, bp_2={self.bp_2}, bp_3={self.bp_3})"

    def __repr__(self):
        return self.__str__()

