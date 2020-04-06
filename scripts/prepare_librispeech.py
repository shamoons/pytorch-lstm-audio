import argparse
import shutil
from tqdm import tqdm
import sys
import os
deepspeech_path = os.path.abspath('./submodules/deepspeech2/data')
sys.path.insert(0, deepspeech_path)
from utils import create_manifest



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_dir', help='Input directory', required=True)

    parser.add_argument(
        '--output_dir', help='Output directory', required=True)
    
    parser.add_argument('--manifest_file', help='Manifest file CSV', required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for root, subdirs, files in tqdm(os.walk(args.input_dir)):
        for f in files:
            if f.find(".flac") != -1:
                full_recording_path = os.path.join(root, f)
                dest_recording_path = os.path.join(args.output_dir, os.path.split(full_recording_path)[1])
            
                
                transcript_file = os.path.join(root, "-".join(f.split('-')[:-1]) + ".trans.txt")
                transcriptions = open(transcript_file).read().strip().split("\n")
                transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]).strip().upper() for t in transcriptions}


                # print(f"full_recording_path: {full_recording_path}\t{f}")
                dest_transcript_path = os.path.join(args.output_dir, os.path.splitext(f)[0] + '.txt')
                shutil.copyfile(full_recording_path, dest_recording_path)

                key = os.path.splitext(f)[0].split("-")[-1]
                
                with open(dest_transcript_path, "w") as f:
                    f.write(transcriptions[key])
                    f.flush()

                # print(transcriptions)


                # assert os.path.exists(full_recording_path) and os.path.exists(root_dir)
                # wav_recording_path = os.path.join(wav_dir, base_filename.replace(".flac", ".wav"))
                # subprocess.call(["sox {}  -r {} -b 16 -c 1 {}".format(full_recording_path, str(args.sample_rate),
                #                                                     wav_recording_path)], shell=True)
                # # process transcript
                # txt_transcript_path = os.path.join(txt_dir, base_filename.replace(".flac", ".txt"))
                # transcript_file = os.path.join(root_dir, "-".join(base_filename.split('-')[:-1]) + ".trans.txt")
                # assert os.path.exists(transcript_file), "Transcript file {} does not exist.".format(transcript_file)
                # transcriptions = open(transcript_file).read().strip().split("\n")
                # transcriptions = {t.split()[0].split("-")[-1]: " ".join(t.split()[1:]) for t in transcriptions}
                # with open(txt_transcript_path, "w") as f:
                #     key = base_filename.replace(".flac", "").split("-")[-1]
                #     assert key in transcriptions, "{} is not in the transcriptions".format(key)
                #     f.write(_preprocess_transcript(transcriptions[key]))
                #     f.flush()
    
    
    create_manifest(args.output_dir, args.manifest_file, audio_extension='flac', skip_order=True)

if __name__ == '__main__':
    main()
