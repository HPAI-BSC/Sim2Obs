import unittest
import matplotlib.pyplot as plt
import numpy as np

from src.data_generation.src.rotate import rotate_vector, get_transform_matrix, rotate_3dvec


# Function to generate points in a spherical distribution
def generate_elliptical_cloud(num_points, a=10, b=5, c=3):
    # Generate random angles
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # azimuthal angle
    costheta = np.random.uniform(-1, 1, num_points)    # cosine of polar angle
    theta = np.arccos(costheta)                         # polar angle

    # Convert spherical coordinates to elliptical Cartesian coordinates
    x = a * np.sin(theta) * np.cos(phi)  # Stretch along x-axis
    y = b * np.sin(theta) * np.sin(phi)  # Stretch along y-axis
    z = c * np.cos(theta)                 # Stretch along z-axis

    return x, y, z


def draw_elliptical_cloud(ax, x, y, z, color='b', label=None):
    # Plot the cloud of points
    ax.scatter(x, y, z, color=color, s=2, label=label)  # s is the size of the points


def draw_rotation_vectors(ax, original_points, rotated_points):
    # Draw arrows to represent the rotation between the original and rotated points
    for orig, rot in zip(original_points, rotated_points):
        ax.quiver(orig[0], orig[1], orig[2],
                   rot[0] - orig[0], rot[1] - orig[1], rot[2] - orig[2],
                   color='k', arrow_length_ratio=0.1)


class TestRotation(unittest.TestCase):
    def test_rotate_vector(self):
        # rotate original axis z around x-axis
        z = np.array([0, 0, 1])
        new_z = rotate_vector(z, axis=np.array([1, 0, 0]), angle_=45)
        print(new_z)

        # Calculate the angle between the original Z-axis and the new Z-axis
        dot_product = np.dot(z, new_z)
        magnitude_v = np.linalg.norm(z)
        magnitude_new_z = np.linalg.norm(new_z)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (magnitude_v * magnitude_new_z)

        # Calculate the angle in radians and then convert to degrees
        theta = np.arccos(cos_theta)
        angle_degrees = np.degrees(theta)

        print("New Z-axis after rotation:", new_z)
        print("Angle between original Z-axis and new Z-axis:", angle_degrees, "degrees")

    def test_get_transform_matrix(self):
        def angle(a, b):
            # Calculate the angle between two vectors
            cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return np.arccos(cos_theta)

        z = np.array([0, 0, 1])
        rotated_z = np.array([0, -0.70710678, 0.70710678])
        Q = get_transform_matrix(rotated_z, z, normalized=True, debug=False)
        print(Q)

        new_vector = Q @ z
        angle_between = angle(z, new_vector)
        print(np.degrees(angle_between))

    def test_rotate(self):
        # rotate original axis z around x-axis
        z = np.array([0, 0, 1])
        new_z = rotate_vector(z, axis=np.array([1, 0, 0]), angle_=45)
        Q = get_transform_matrix(new_z, z, normalized=True, debug=False)

        # Generate elliptical points
        x, y, z = generate_elliptical_cloud(num_points=100, a=10, b=5, c=3)

        # Create a new figure for 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw original cloud
        draw_elliptical_cloud(ax, x, y, z, color='b', label='Original Cloud')

        # Rotate the cloud
        cloud = np.float32([elem for elem in zip(x, y, z)])
        rotated_cloud = np.float32(rotate_3dvec(Q, cloud))
        x_rotated, y_rotated, z_rotated = rotated_cloud[:, 0], rotated_cloud[:, 1], rotated_cloud[:, 2]

        # Draw rotated cloud in the same figure
        draw_elliptical_cloud(ax, x_rotated, y_rotated, z_rotated, color='r', label='Rotated Cloud')

        # Draw rotation vectors
        draw_rotation_vectors(ax, cloud, rotated_cloud)

        # Set labels for axes
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # Set the title
        ax.set_title('3D Elliptical Cloud of Points and Rotated Cloud Visualization')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()

    def test_rotate_z(self):
        # rotate original axis z around x-axis
        z = np.array([0, 0, 1])
        new_z = rotate_vector(z, axis=np.array([0, 0, 1]), angle_=270)
        Q = get_transform_matrix(new_z, z, normalized=True, debug=False)

        # Generate elliptical points
        x, y, z = generate_elliptical_cloud(num_points=100, a=10, b=5, c=3)

        # Create a new figure for 3D plotting
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Draw original cloud
        draw_elliptical_cloud(ax, x, y, z, color='b', label='Original Cloud')

        # Rotate the cloud
        cloud = np.float32([elem for elem in zip(x, y, z)])
        rotated_cloud = np.float32(rotate_3dvec(Q, cloud))
        x_rotated, y_rotated, z_rotated = rotated_cloud[:, 0], rotated_cloud[:, 1], rotated_cloud[:, 2]

        # Draw rotated cloud in the same figure
        draw_elliptical_cloud(ax, x_rotated, y_rotated, z_rotated, color='r', label='Rotated Cloud')

        # Draw rotation vectors
        draw_rotation_vectors(ax, cloud, rotated_cloud)

        # Set labels for axes
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        # Set the title
        ax.set_title('3D Elliptical Cloud of Points and Rotated Cloud Visualization')

        # Add a legend
        ax.legend()

        # Show the plot
        plt.show()