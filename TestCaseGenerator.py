import os
from shutil import copyfile
import random
from tqdm import tqdm

# Number of writers to generate the test dataset from
number_of_writers = 672

# Number of test cases needed
number_of_test_cases = 600

# Path for Writers data
writers_path = "Data\Writers\\"

# Path to generate the dataset at
test_dataset_directory = "Data\data2\\"

# Path to all of the images in the dataset
all_dataset_directory = "Data\AllDataSet\\"

# Path to the writers form.txt
writers_form_directory = "Data\\asciiData\\forms.txt"


def get_writers_images_names(file_path):
    try:
        file = open(file_path, "r")
        file_lines = file.readlines()
        for line_number in range(len(file_lines)):
            file_lines[line_number] = file_lines[line_number].rstrip("\n")
        file.close()
        return file_lines
    except IOError:
        return []


# This cell is responsible for grouping the ids of a specific writer and discarding writers who has less than 3 images
writer_images_id = []
for i in range(number_of_writers):
    writerID = '%03d' % i
    writerImagesIDItem = get_writers_images_names(writers_path + writerID + ".txt")
    if len(writerImagesIDItem) >= 3:
        writer_images_id.append(writerImagesIDItem)

# This function is responsible for getting the writed id of a specific image name


def get_writer_id(path, image_id):

    image_id = image_id.split(".")[0]
    file = open(path, "r")
    file_lines = file.readlines()
    image_id = image_id + " "
    for line in file_lines[16:]:
        if image_id in line:
            if line[7] == " ":
                return line[8:11]
            else:
                return line[9:12]
    file.close()


# Generating the folders if not available and create them
for i in range(number_of_test_cases):
    z = '%02d' % i
    directory = test_dataset_directory + z + '\\'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if not os.path.exists(directory+'1\\'):
        os.makedirs(directory+'1\\')
    if not os.path.exists(directory+'2\\'):
        os.makedirs(directory+'2\\')
    if not os.path.exists(directory+'3\\'):
        os.makedirs(directory+'3\\')

for i in tqdm(range(number_of_test_cases)):
    test_case_number = '%02d' % i
    directory = test_dataset_directory + test_case_number + '\\'
    random_writer_index1 = random.randrange(0, len(writer_images_id), 1)

    writers_id = []

    # Generating images of writer 1 in the test case
    image1 = ''
    image2 = ''
    while image1 == image2:
        image1 = writer_images_id[random_writer_index1][random.randrange(
            0, len(writer_images_id[random_writer_index1]), 1)]
        image2 = writer_images_id[random_writer_index1][random.randrange(
            0, len(writer_images_id[random_writer_index1]), 1)]
    destination1 = test_dataset_directory + test_case_number + '\\1\\' + image1
    destination2 = test_dataset_directory + test_case_number + '\\1\\' + image2
    copyfile('Data\\AllDataSet\\'+image1, destination1)
    copyfile('Data\\AllDataSet\\'+image2, destination2)
    writers_id.append(get_writer_id(writers_form_directory, image1))

    # Generating images of writer 2 in the test case
    random_writer_index2 = random.randrange(0, len(writer_images_id), 1)
    while random_writer_index2 == random_writer_index1:
        random_writer_index2 = random.randrange(0, len(writer_images_id), 1)

    image3 = ''
    image4 = ''
    while image3 == image4:
        image3 = writer_images_id[random_writer_index2][random.randrange(
            0, len(writer_images_id[random_writer_index2]), 1)]
        image4 = writer_images_id[random_writer_index2][random.randrange(
            0, len(writer_images_id[random_writer_index2]), 1)]
    destination1 = test_dataset_directory + test_case_number + '\\2\\' + image3
    destination2 = test_dataset_directory + test_case_number + '\\2\\' + image4
    copyfile(all_dataset_directory + image3, destination1)
    copyfile(all_dataset_directory + image4, destination2)
    writers_id.append(get_writer_id(writers_form_directory, image3))

    # Generating images of writer 3 in the test case
    random_writer_index3 = random.randrange(0, len(writer_images_id), 1)
    while random_writer_index3 == random_writer_index1 or random_writer_index3 == random_writer_index2:
        random_writer_index3 = random.randrange(0, len(writer_images_id), 1)

    image5 = ''
    image6 = ''
    while image5 == image6:
        image5 = writer_images_id[random_writer_index3][random.randrange(
            0, len(writer_images_id[random_writer_index3]), 1)]
        image6 = writer_images_id[random_writer_index3][random.randrange(
            0, len(writer_images_id[random_writer_index3]), 1)]
    destination1 = test_dataset_directory + test_case_number + '\\3\\' + image5
    destination2 = test_dataset_directory + test_case_number + '\\3\\' + image6
    copyfile(all_dataset_directory + image5, destination1)
    copyfile(all_dataset_directory + image6, destination2)
    writers_id.append(get_writer_id(writers_form_directory, image5))

    # Generating the test image randomly from the dataset of the 3 writers
    # Making sure the test image is different than those from the given training images
    random_writer_index_test = random.choice(
        [random_writer_index1, random_writer_index2, random_writer_index3])
    imageTest = writer_images_id[random_writer_index_test][random.randrange(
        0, len(writer_images_id[random_writer_index_test]), 1)]
    while imageTest == image1 or imageTest == image2 or imageTest == image3 or imageTest == image4 or \
            imageTest == image5 or imageTest == image6:
        imageTest = writer_images_id[random_writer_index_test][random.randrange(
            0, len(writer_images_id[random_writer_index_test]), 1)]
    copyfile(all_dataset_directory + imageTest,
             test_dataset_directory + test_case_number + '\\' + imageTest)
    writers_id.append(get_writer_id(writers_form_directory, imageTest))

    f = open(test_dataset_directory + test_case_number + "\\ids.txt", "a")
    for id_index in range(len(writers_id)):
        f.write(writers_id[id_index] + "\n")
    f.close()
