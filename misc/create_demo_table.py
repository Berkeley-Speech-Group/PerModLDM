import os
import shutil 
import numpy as np 
import pandas as pd 

## create_demo_table.py 
# 
# This file generates in the table for the demo, and copies in the files to the paths as necessary

# Searces for appropriate labelling and returns the right text
def getLetter(file_name):
    if "SJ" in file_name or "PT" in file_name:
        return "A"
    else:
        return "T"

# Format the filename for extraction 
def formatFile(file_name):
    letter = getLetter(file_name)
    if letter == "A":
        form_file = file_name.split("_")[0] 
    else:
        # Transform the file to be correct
        if "16k" not in file_name:
            form_file = file_name.split(".")[0]
            form_file = form_file + "_16k"
        else:
            form_file = file_name.split(".")[0]

    return form_file



sources = ["p293_386_mic1.flac", "p317_188_mic1_16k.flac", "p259_277_mic1_16k.flac", "PT019_eggs.wav", "PT008_hit.wav"]
source_letter = [getLetter(x) for x in sources]


ground_truth_files = os.listdir("../resources/audios/ground_truth")

# Generate all the paths
gt_path = "../resources/audios/ground_truth"
audio_path = "../resources/audios/"
local_dsvae = os.path.join(audio_path, "dsvae")
local_pretrained = os.path.join(audio_path, "pretrained")
local_finetuned = os.path.join(audio_path, "finetuned")

ext_audio_path = "/data/robbizorg"
ext_samples = "/data/robbizorg/permod_eval_gens/audio_quality_vctk"
ext_dsvae = os.path.join(ext_samples, "dsvae_wavs")
ext_permod = os.path.join(ext_samples, "demo_wavs")

# Loop through and get the examples per
demo_dict = {
    "task": [],
    "source": [],
    "target": [],
    "dsvae": [],
    "pretrained": [],
    "finetuned": []
}

for source in sources:
    for gt in ground_truth_files:
        # Skip over duplicates 
        if source == gt:
            continue

        form_source = formatFile(source)
        form_target = formatFile(gt)
        # Copy over the appropriate files

        # Just take the first sample from DSVAE/PerMod
        dsvae_file = "dsvae_target_" + form_target + "_source_" + form_source + "_00.wav"
        pretrained_file = "vctkrf_target_" + form_target + "_source_" + form_source + "_00.wav" 
        finetuned_file = "vctkrf_finetuned_target_" + form_target + "_source_" + form_source + "_00.wav" 

        task = getLetter(source) + "2" + getLetter(gt)

        demo_dict["task"].append(task)
        demo_dict["source"].append(form_source)
        demo_dict["target"].append(form_target)
        demo_dict["dsvae"].append(dsvae_file)
        demo_dict["pretrained"].append(pretrained_file)
        demo_dict["finetuned"].append(finetuned_file)

        ## Copy over files if they don't exist
        if not os.path.exists(os.path.join(local_dsvae, dsvae_file)):
            shutil.copyfile(os.path.join(ext_dsvae, dsvae_file), os.path.join(local_dsvae, dsvae_file)) 

        if not os.path.exists(os.path.join(local_pretrained, pretrained_file)):
            shutil.copyfile(os.path.join(ext_permod, pretrained_file), os.path.join(local_pretrained, pretrained_file)) 

        if not os.path.exists(os.path.join(local_finetuned, finetuned_file)):
            shutil.copyfile(os.path.join(ext_permod, finetuned_file), os.path.join(local_finetuned, finetuned_file)) 

        
demo_df = pd.DataFrame(demo_dict)

# Filter out the ones with SJ5002 since that is a pretty typical voice (low severity)
demo_df = demo_df[demo_df["target"] != "SJ5002"]

table_string = ""

task_list = ["Typical-to-Typical (T2T)", "Typical-to-Aypical (T2A)", "Atypical-to-Typical (A2T)", "Atypical-to-Atypical (A2A)"]

for task_name in task_list:
    task_sub_name = task_name.split("(")[-1][:-1]
    table_string += """<center><h2>%s</h2></center>
<tr>
	<th>Input Speech</th>
	<th>Target Speech</th>
	<th>DSVAE (VC)</th>
	<th>PerMod-Pretrained (VM)</th>
	<th>PerMod-Finetuned (VM)</th>
</tr>\n\n""" % task_name

    sub_df = demo_df[demo_df["task"] == task_sub_name]

    for row in sub_df.iterrows():
        source = os.path.join(gt_path, row[1]["source"])
        target = os.path.join(gt_path, row[1]["target"]) 
        dsvae = os.path.join(local_dsvae, row[1]["dsvae"])
        pretrained = os.path.join(local_pretrained, row[1]["pretrained"])
        finetuned = os.path.join(local_finetuned, row[1]["finetuned"])

        table_string += """<tr>
	<td><audio controls><source src="%s" type="audio/flac"></audio></td>
	<td><audio controls><source src="%s" type="audio/flac"></audio></td>
	<td><audio controls><source src="%s" type="audio/wav"></audio></td>
	<td><audio controls><source src="%s"></audio></td>
	<td><audio controls><source src="%s" type="audio/wav"></audio></td>
</tr>\n\n""" % (source, target, dsvae, pretrained, finetuned)

with open("./table.html", "w") as table_file:
    table_file.write(table_string)

