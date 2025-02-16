# Developed for the Vera C. Rubin Observatory/LSST Data Management System.
# This product includes software developed by the
# Vera C. Rubin Observatory/LSST Project (https://www.lsst.org).
#
# Copyright 2020 University of Washington
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import importlib_resources

from numba import njit

deg2rad = np.radians
sin = np.sin
cos = np.cos

logger = logging.getLogger(__name__)


def distToSegment(points, x0, y0, x1, y1):
    """Compute the distance from each point to the line segment defined by
    the points (x0, y0) and (x1, y1).  Returns the distance in the same
    units as the points are specified in (radians, degrees, etc.). Uses planar
    geometry for the calculations (assuming small angular distances).

    Parameters
    -----------
    points : array
        Array of shape (2, n) describing the corners of the sensor.

    x0 : float
        The x coordinate of the first end of the segment.

    y0 : float
        The y coordinate of the first end of the segment.

    x1 : float
        The x coordinate of the second end of the segment.

    y1 : float
        The y coordinate of the second end of the segment.

    Returns
    --------
    dist : array
        Array of length n storing the distances.
    """
    # Handle the case where the segment is a point: (x0 == x1) and (y0 == y1)
    len_sq = (x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0)
    if len_sq == 0.0:
        return np.sqrt((points[0] - x0) * (points[0] - x0) + (points[1] - y0) * (points[1] - y0))

    # Find the closest point on the line segment.
    t = ((points[0] - x0) * (x1 - x0) + (points[1] - y0) * (y1 - y0)) / len_sq
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    proj_x = x0 + (x1 - x0) * t
    proj_y = y0 + (y1 - y0) * t

    # Compute the distances to the closest points on the line segment.
    return np.sqrt((points[0] - proj_x) * (points[0] - proj_x) + (points[1] - proj_y) * (points[1] - proj_y))


# ==============================================================================
# coordinate transforms
# ==============================================================================


def radec_to_tangent_plane(ra, dec, field_ra, field_dec):
    """
    Converts ra and dec to xy on the plane tangent to image center, in the 2-d coordinate system where y is aligned with the meridian.

    Parameters:
    -----------
    ra (float/array of floats): observation Right Ascension, radians.

    dec (float/array of floats): observation Declination, radians.

    fieldra (float/array of floats): field pointing Right Ascension, radians.

    fielddec (float/array of floats): field pointing Declination, radians.

    fieldID (float/array of floats): Field ID, optional.

    Returns:
    ----------
    x, y (float/array of floats): Coordinates on the focal plane, radians projected
    to the plane tangent to the unit sphere.

    """

    # convert to cartesian coordiantes on unit sphere
    observation_vectors = np.array([cos(ra) * np.cos(dec), sin(ra) * np.cos(dec), sin(dec)])  # x  # y  # z

    field_vectors = np.array(
        [cos(field_ra) * np.cos(field_dec), sin(field_ra) * np.cos(field_dec), sin(field_dec)]
    )

    # make the basis vectors for the fields of view
    # the "x" basis is easy, 90 d rotation of the x, y components
    focalx = np.zeros(field_vectors.shape)
    focalx[0] = -field_vectors[1]
    focalx[1] = field_vectors[0]

    # "y" by taking cross product of field vector and "x"
    focaly = np.cross(field_vectors, focalx, axis=0)

    # normalize
    focalx /= np.linalg.norm(focalx, axis=0)
    focaly /= np.linalg.norm(focaly, axis=0)

    # extend observation vectors to plane tangent to field pointings
    k = 1.0 / np.sum(field_vectors * observation_vectors, axis=0)
    observation_vectors *= k
    observation_vectors -= field_vectors

    # get observation vectors as combinations of focal vectors
    x = np.sum(observation_vectors * focalx, axis=0)
    y = np.sum(observation_vectors * focaly, axis=0)

    return x, y


def radec_to_focal_plane(ra, dec, field_ra, field_dec, field_rot):
    # convert ra, dec to points on focal plane, x pointing to celestial north
    x, y = radec_to_tangent_plane(ra, dec, field_ra, field_dec)
    # rotate focal plane to align with detectors
    xy = x + 1.0j * y
    xy *= np.exp(1.0j * field_rot)  # which direction to rotate?

    x = np.real(xy)
    y = np.imag(xy)

    return x, y


# ==============================================================================
# detector class
# ==============================================================================


class Detector:
    """Detector class"""

    def __init__(self, points, ID=0, units="radians"):
        """
        Initiates a detector object.

        Parameters
        -----------
        points : array
            Array of shape (2, n) describing the corners of the sensor.

        ID : int, optional
            Aan integer ID for the sensor. Default =0.

        units : string, optional
            Units that points is provided in, "radians" or "degrees" from
            center of the focal plane. Default = "radians"

        Returns
        ----------
        detector : Detector
            A Detector object instance.

        """

        # points  --->   should be shape dims, n points
        self.ID = ID
        self.ra = points[0]
        self.dec = points[1]
        self.units = units

        if units == "degrees" or units == "deg":
            self.deg2rad()

        # generate focal plane coordinates
        # convert to xyz on unit sphere
        z = np.cos(self.ra) * np.cos(self.dec)  # x
        self.x = sin(self.ra) * np.cos(self.dec)  # y
        self.y = sin(self.dec)

        # project to focal plane
        self.x /= z
        self.y /= z

        # calculate centroid
        self.centerx = np.sum(self.x) / len(self.x)
        self.centery = np.sum(self.y) / len(self.y)

    def ison(self, point, ϵ=10.0 ** (-11), edge_thresh=None, plot=False):
        """
        Determines whether a point (or array of points) falls on the
        detector.

        Parameters
        -----------
        point : array
            Array of shape (2, n) for n points.

        ϵ : float, optional
            Threshold for whether point is on detector. Default: 10.0 ** (-11)

        edge_thresh: float, optional
            The focal plane distance (in arcseconds) from the detector's edge
            for a point to be counted. Removes points that are too
            close to the edge for source detection. Default = None

        plot : Boolean, optional
            Flag for whether to plot the detector and the point. Default = False

        Returns
        ----------
        selectedidx : array
            Indices of points in point array that fall on the sensor.
        """
        # points needs to be shape 2,n
        # if single point, needs to be an array of single element arrays
        if len(point.shape) != 2 or point.shape[0] != 2:
            logger.error(f"ERROR: Detector.ison invalid array {point.shape}")
            sys.exit(f"ERROR: Detector.ison invalid array {point.shape}")

        # check whether point is in circle bounding the detector
        r2 = np.max((self.x - self.centerx) ** 2 + (self.y - self.centery) ** 2)
        selectedidx = np.where((point[0] - self.centerx) ** 2 + (point[1] - self.centery) ** 2 <= r2)[0]

        # check whether selected fall on the detector
        # compare true area to the segmented area
        selectedidx = selectedidx[np.abs(self.segmentedArea(point[:, selectedidx]) - self.trueArea()) <= ϵ]

        # If there is a threshold to the edge, further filter selectedidx.
        if edge_thresh is not None and len(selectedidx) > 0:
            # Convert edge threshold to the same units as the detector.
            if self.units == "degrees" or self.units == "deg":
                edge_thresh = edge_thresh / 3600.0
            elif self.units == "radians" or self.units == "rad":
                edge_thresh = np.radians(edge_thresh / 3600.0)
            else:
                logger.error(f"ERROR: Detector.ison unable to convert edge_thresh to {self.units}")
                sys.exit(f"ERROR: Detector.ison unable to convert edge_thresh to {self.units}")

            n = len(self.x)
            for i in range(n):  # test each edge
                dist_to_edge = distToSegment(
                    point[:, selectedidx],
                    self.x[i],
                    self.y[i],
                    self.x[(i + 1) % n],
                    self.y[(i + 1) % n],
                )
                selectedidx = selectedidx[dist_to_edge > edge_thresh]

        if plot:
            x = point[0][selectedidx]
            y = point[1][selectedidx]

            plt.scatter(x, y, color="red", s=3.0)

        return selectedidx

    def trueArea(self):
        """
        Returns the area of the detector. Uses the same method as
        segmentedArea, but the test point is the mean of the corner coordinates.
        Will probably fail if the sensor is not convex.

        Parameters
        -----------
        None.

        Returns
        ----------
        area : float
            The area of the detector.

        """
        x = self.x - self.centerx
        y = self.y - self.centery

        xrolled = np.roll(x, 1)
        yrolled = np.roll(y, 1)

        area = 0.5 * np.sum(np.abs(x * yrolled - y * xrolled))

        return area

    def segmentedArea(self, point):
        """
        Returns the area of the detector by calculating the area of each
        triangle segment defined by each pair of adjacent corners and a point
        inside the sensor.
        Fails if the point is not inside the sensor or if the sensor is not
        convex.

        Parameters
        -----------
        point : array
            Array of shape (2, n) for n points.

        Returns
        ----------
        area : float
            The area of the detector.
        """

        # so that both a single and many points work
        ncorners = self.x.shape[0]

        if len(point.shape) == 1:
            x = self.x - point[0]
            y = self.y - point[1]

        else:
            x = (np.zeros((ncorners, point.shape[1])).T + self.x).T - point[0]
            y = (np.zeros((ncorners, point.shape[1])).T + self.y).T - point[
                1
            ]  # copy over an array to make broadcasting work

        xrolled = np.roll(x, 1, axis=0)
        yrolled = np.roll(y, 1, axis=0)
        area = []

        for i in range(len(self.x)):
            area.append(np.abs(x[i] * yrolled[i] - y[i] * xrolled[i]))

        return 0.5 * (np.sum(area, axis=0))

    def sortCorners(self):
        """
        Sorts the corners to be counterclockwise by angle from center of
        the detector. Modifies self.

        Parameters
        -----------
        None.

        Returns
        ----------
        None.

        """

        # convert corners to angles (radians)
        theta = np.arctan2(self.y - self.centery, self.x - self.centerx)

        neworder = np.argsort(theta)
        self.x = self.x[neworder]
        self.y = self.y[neworder]

    def rotateDetector(self, theta):
        """
        Rotates a sensor around the origin of the coordinate system its
        corner locations are provided in.

        Parameters
        -----------
        theta : float
            Angle to rotate by, in radians.

        Returns
        ----------
        Detector:  Detector
            New Detector instance.

        """

        # convert rotation angle to complex number
        q = cos(theta) + sin(theta) * 1.0j

        # convert points to complex numbers
        coords = self.x + self.y * 1.0j

        # rotate
        newcoords = coords * q
        return Detector(np.array((np.real(newcoords), np.imag(newcoords))), self.ID)

    def rad2deg(self):
        """
        Converts corners from radians to degrees.

        Parameters
        -----------
        None.

        Returns
        ----------
        None.

        """

        if self.units == "radians":
            self.x = np.degrees(self.x)
            self.y = np.degrees(self.y)
            self.centerx = np.degrees(self.centerx)
            self.centery = np.degrees(self.centery)
            self.units = "degrees"
        else:
            print("Units are already degrees")

    def deg2rad(self):
        """
        Converts corners from degrees to radians.

        Parameters
        -----------
        None.

        Returns
        ----------
        None.

        """

        if self.units == "degrees":
            self.x = np.radians(self.x)
            self.y = np.radians(self.y)
            self.centerx = np.radians(self.centerx)
            self.centery = np.radians(self.centery)
            self.units = "radians"
        else:
            print("Units are already radians")

    def plot(self, theta=0.0, color="gray", units="rad", annotate=False):
        """
        Plots the footprint for an individual sensor. Currently not on the
        focal plane, just the sky coordinates. Relatively minor difference
        (width of footprint for LSST is <2.1 degrees), so should be fine for
        internal demonstration purposes, but not for confirming algorithms or
        for offical plots.

        Parameters
        -----------
        theta : float, optional
            Aangle to rotate footprint by, radians or degrees. Default =0.0

        color :string, optional
            Line color. Default = "gray"

        units: string, optional
            Units. Units is provided in ("deg" or "rad"). Default = 'rad'.

        annotate : Boolean
            Flag whether to annotate each sensor with its index in self.detectors.
            Default = False

        Returns
        ----------
        None.

        """

        detector = self.rotateDetector(theta)
        if units == "deg":
            detector.rad2deg()
        nd = len(self.x)
        x = np.zeros(nd + 1)
        x[0:nd] = detector.x
        x[-1] = x[0]
        y = np.zeros(nd + 1)
        y[0:nd] = detector.y
        y[-1] = y[0]
        plt.plot(x, y, color=color)

        if annotate is True:
            plt.annotate(str(detector.ID), (detector.centerx, detector.centery))


# ==============================================================================
# camera class
# ==============================================================================


class Footprint:
    """Camera footprint class"""

    def __init__(self, path=None, detectorName="detector"):
        """
        Initiates a Footprint object.

        Parameters
        -----------
        path : string, optional
            Path to a .csv file containing detector corners. Default = None

        detectorName : string, optional
            Name of column in detector file indicating to which sensor a
            corner belongs. Default = "detector"

        Returns
        ----------
        Footprint : Footprint
            Footprint object for the provided sensors.

        """

        # file should be a .csv (and should be actually comma seperated)
        # the center of the camera should be the origin
        # if the user doesn't provide their own version of the footprint,
        # we'll use the default LSST version that comes included.
        if path:
            try:
                allcornersdf = pd.read_csv(path)
                logger.info(f"Using CCD Detector file: {path}")
            except IOError:
                logger.error(f"Provided detector footprint file does not exist.")
                sys.exit(1)

        else:
            try:
                default_camera_config_file = "data/LSST_detector_corners_100123.csv"
                # stream = pkg_resources.resource_stream(__name__, default_camera_config_file)
                # stream = importlib_resources.as_file( default_camera_config_file )
                stream = importlib_resources.files(__name__).joinpath(default_camera_config_file)
                logger.info(f"Using built-in CCD Detector file: {default_camera_config_file}")
                allcornersdf = pd.read_csv(stream)
            except IOError:
                logger.error(f"Error loading default camera footprint, exiting ...")
                sys.exit(1)

        # build dictionary of detectorName:[list_of_inds]
        det_to_inds = {}
        for det in allcornersdf.detector.unique():
            det_to_inds[det] = allcornersdf.index[allcornersdf.detector == det].tolist()

        # create a list of `Detector` objects using the list_of_inds for each detector
        self.detectors = [
            Detector(
                np.array(
                    (
                        allcornersdf.iloc[inds].x,
                        allcornersdf.iloc[inds].y,
                    )
                ),
                det,
            )
            for det, inds in det_to_inds.items()
        ]

        self.N = len(self.detectors)

        # sort the corners of each detector
        for i in range(self.N):
            self.detectors[i].sortCorners()

    def plot(self, theta=0.0, color="gray", units="rad", annotate=False):
        """
        Plots the footprint. Currently not on the focal plane, just the sky
        coordinates. Relatively minor difference (width of footprint for LSST
        is <2.1 degrees), so should be fine for internal demonstration
        purposes, but not for confirming algorithms or for offical plots.

        Parameters
        -----------
        theta : float, optional
            Angle to rotate footprint by, radians or degrees. Default = 0.0

        color : string, optional
            Line color. Default = "gray"

        units : string, optional
            Units theta is provided in ("deg" or "rad"). Default = "rad"

        annotate : boolean, optional
            Whether to annotate each sensor with its index in
            self.detectors. Default = False

        Returns
        ----------
        None.

        """

        for i in range(self.N):
            self.detectors[i].plot(theta=theta, color=color, units=units, annotate=annotate)

    def applyFootprint(
        self,
        field_df,
        ra_name="RA_deg",
        dec_name="Dec_deg",
        field_name="FieldID",
        ra_name_field="fieldRA_deg",
        dec_name_field="fieldDec_deg",
        rot_name_field="fieldRotSkyPos_deg",
        edge_thresh=None,
    ):
        """
        Determine whether detections fall on the sensors defined by the
        footprint. Also returns the an ID for the sensor a detection is made
        on.

        Parameters
        -----------
        field_df : Pandas dataframe
            Dataframe containing detection information with pointings.

        ra_name : string, optional
            "field_df" dataframe's column name for object's RA
             for the given observation. Default = "RA_deg" [units: degrees]

        dec_name : string, optional
            "field_df" dataframe's column name for object's declination
             for the given observation. Default = "Dec_deg" [units: dgrees]

        ra_name_field : string, optional
            "field_df" dataframe's column name for the observation field's RA
             Default = "fieldRA_deg" [units: degrees]

        dec_name_field : string, optional
            "field_df" dataframe's column name for the observation field's declination
             Default = "fieldDec_deg" [Units: degrees]

        rot_name_field: string, optional
            "field_df" dataframe's column name for the observation field's rotation angle
            Default = "fieldRotSkyPos_deg" [Units: degrees]

        edge_thresh: float, optional
            An angular threshold in arcseconds for dropping pixels too close to the edge.
            Default  = None

        Returns
        ----------
        detected : array
            Indices of rows in field_df which fall on the sensor(s).

        detectorID : array
            Index corresponding to a detector in self.detectors for each entry in detected.

        """

        # convert detections to xyz on unit sphere
        ra = deg2rad(field_df[ra_name])
        dec = deg2rad(field_df[dec_name])

        # convert field pointings to xyz on unit sphere
        fieldra = deg2rad(field_df[ra_name_field])
        fielddec = deg2rad(field_df[dec_name_field])
        rotSkyPos = deg2rad(field_df[rot_name_field])

        # (no rotation on 3d unit sphere):
        points = np.array((radec_to_focal_plane(ra, dec, fieldra, fielddec, rotSkyPos)))
        # x, y = radec_to_focal_plane(ra, dec, fieldra, fielddec, rotSkyPos)
        # points = np.array((x, y))

        # check whether they land on any of the detectors
        detected = []
        detectorId = []
        i = 0
        for detector in self.detectors:
            stuff = detector.ison(points, edge_thresh=edge_thresh)
            detected.append(stuff)
            # detectorId.append([detector.ID] * len(stuff))
            detectorId.append([i] * len(stuff))
            i += 1

        return np.concatenate(detected), np.concatenate(detectorId)
