import os
import shutil
from pathlib import Path
import imageio
import numpy as np

from plotting import plot_district_map

def make_animation(title, assignments):
    """
    Creates a .gif file with title 'title', using the Utah district assignments
    specified. In the process creates and deletes a directory called source_images.
    The file is created in the directory Animations/

    Parameters:
        title (str): title of the output file
        assignments (pd.DataFrame): Dataframe where each row is a districting assignment
            (i. e. a mapping from precinct ids to district assignments)
    """
    # Create our intermediate directory for storing the component .png images
    shutil.rmtree('source_images')
    os.mkdir('source_images')

    # Create district maps of each assignment
    for i in range(len(assignments)):
        a = np.array(assignments.loc[i, :])
        plot_district_map(a, save=True, savetitle='source_images/{}.png'.format(i))

    # Gather the appropriate paths
    image_path = Path('source_images')
    images = list(image_path.glob('*'))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))

    # Write the .gif
    imageio.mimwrite('Animations/{}'.format(title), image_list)

    # Delete the intermediate directory source_images
    shutil.rmtree('source_images')
