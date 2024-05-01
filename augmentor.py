import os
from os import path, listdir
from subprocess import run, DEVNULL

sketch_root = '/home/aidan/ComputerScience/school/18786-DeepLearning/project/data/256x256/sketch/tx_000100000000/'
real_root = '/home/aidan/ComputerScience/school/18786-DeepLearning/project/data/256x256/photo/tx_000100000000/'
tmp_path = '/tmp/sketchy_tmp/'
example_path = "/home/aidan/ComputerScience/school/18786-DeepLearning/project/photosketch/PhotoSketch-master/examples/"
# test = '/tmp/sketchy_test'

def main():
    os.environ['sketchHomeDir'] = tmp_path # define this as output dir for the script
    if not path.exists(tmp_path):
        print(f'Creating temp {tmp_path}')
        run(['mkdir', tmp_path], check=True)
    else:
        print(f'Temp {tmp_path} already exists.')

    for directory in listdir(real_root):
        print(f"Decending into {directory}...")

        run([f'rm -f {example_path}*'], shell=True, check=True)
        run([f'rm -f {tmp_path}*'], shell=True, check=True)

        joined_path = path.join(real_root, directory)
        if not path.isdir(joined_path):
            print(f'Found non-dir {joined_path}')
            continue

        run([f'cp {joined_path}/* {example_path}'], shell=True, check=True) # put photo in exmpl dir
        run('./run_augmentor.sh', shell=True, check=True, stdout=DEVNULL) # create sketches

        joined_sketch_path = path.join(sketch_root, directory)
        print('\tCopying files...')
        for sketch in listdir(tmp_path):
            prefix, suffix = sketch.split('.')
            new_name = prefix + '-99.' + suffix

            sketch_path = path.join(tmp_path, sketch) # take files from tmp: rename them, and add them
            run([f'mv {sketch_path} {joined_sketch_path}/{new_name}'], shell=True, check=True)


if __name__ == '__main__':
    print('===Warning, this program will wipe all data in Photosketch-master/examples!===')
    if input('Continue? (y/N): ') == 'y':
        main()
    print("Finished. Exiting...")
