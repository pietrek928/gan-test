import os
import random
from argparse import Namespace
from importlib import reload
from urllib.request import urlretrieve

import PIL
import cv2
import kornia.augmentation as K
import noise
import numpy as np
from PIL import Image
from torch import nn

import models
from dirs import get_path
from download import download_all
from models import add_noise, ModelHost

reload(PIL.TiffTags)

download_all()

# @markdown >`prompts` is the list of prompts to give to the AI, separated by `|`. With more than one, it will attempt to mix them together. You can add weights to different parts of the prompt by adding a `p:x` at the end of a prompt (before a `|`) where `p` is the prompt and `x` is the weight.
prompts = "A fantasy landscape, by Greg Rutkowski. A lush mountain.:1 | Trending on ArtStation, unreal engine. 4K HD, realism.:0.63"  # @param {type:"string"}
width = 480  # @param {type:"number"}
height = 480  # @param {type:"number"}
model = "ImageNet 16384"  # @param ['ImageNet 16384', 'ImageNet 1024', "Gumbel 8192", "Sber Gumbel", 'WikiArt 1024', 'WikiArt 16384', 'WikiArt 7mil', 'COCO-Stuff', 'COCO 1 Stage', 'FacesHQ', 'S-FLCKR']

if model == "Gumbel 8192" or model == "Sber Gumbel":
    is_gumbel = True
else:
    is_gumbel = False

##@markdown The flavor effects the output greatly. Each has it's own characteristics and depending on what you choose, you'll get a widely different result with the same prompt and seed. Ginger is the default, nothing special. Cumin results more of a painting, while Holywater makes everythng super funky and/or colorful. Custom is a custom flavor, use the utilities above.
#   Type "old_holywater" to use the old holywater flavor from Hypertron V1
flavor = 'ginger'  # @param ["ginger", "cumin", "holywater", "zynth", "wyvern", "aaron", "moth", "juu", "custom"]
template = 'Balanced'  # @param ["none", "----------Parameter Tweaking----------", "Balanced", "Detailed", "Consistent Creativity", "Realistic", "Smooth", "Subtle MSE", "Hyper Fast Results", "----------Complete Overhaul----------", "flag", "planet", "creature", "human", "----------Sizes----------", "Size: Square", "Size: Landscape", "Size: Poster", "----------Prompt Modifiers----------", "Better - Fast", "Better - Slow", "Movie Poster", "Negative Prompt", "Better Quality"]
##@markdown To use initial or target images, upload it on the left in the file browser. You can also use previous outputs by putting its path below, e.g. `batch_01/0.png`. If your previous output is saved to drive, you can use the checkbox so you don't have to type the whole path.
init = 'default noise'  # @param ["default noise", "image", "random image", "salt and pepper noise", "salt and pepper noise on init image"]
init_image = ""  # @param {type:"string"}

if init == "random image":
    url = "https://picsum.photos/" + str(width) + "/" + str(height) + "?blur=" + str(random.randrange(5, 10))
    init_image = get_path("Init_Img/Image.png")
    urlretrieve(url, init_image)
elif init == "random image clear":
    url = "https://source.unsplash.com/random/" + str(width) + "x" + str(height)
    init_image = get_path("Init_Img/Image.png")
    urlretrieve(url, init_image)
elif init == "random image clear 2":
    url = "https://loremflickr.com/" + str(width) + "/" + str(height)
    init_image = get_path("Init_Img/Image.png")
    urlretrieve(url, init_image)
elif init == "salt and pepper noise":
    init_image = get_path("Init_Img/Image.png")
    urlretrieve("https://i.stack.imgur.com/olrL8.png", init_image)
    img = cv2.imread(init_image, 0)
    cv2.imwrite(init_image, add_noise(img))
elif init == "salt and pepper noise on init image":
    img = cv2.imread(init_image, 0)
    init_image = get_path("Init_Img/Image.png")
    cv2.imwrite(init_image, add_noise(img))
elif init == "perlin noise":
    # For some reason Colab started crashing from this
    shape = (width, height)
    scale = 100
    octaves = 6
    persistence = 0.5
    lacunarity = 2.0
    seed = np.random.randint(0, 100000)
    world = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            world[i][j] = noise.pnoise2(i / scale, j / scale, octaves=octaves, persistence=persistence,
                                        lacunarity=lacunarity, repeatx=1024, repeaty=1024, base=seed)
    init_image = get_path("Init_Img/Image.png")
    Image.fromarray(prep_world(world)).convert("L").save(init_image)
elif init == "black and white":
    url = "https://www.random.org/bitmaps/?format=png&width=300&height=300&zoom=1"
    init_image = get_path("Init_Img/Image.png")
    urlretrieve(url, init_image)

seed = -1  # @param {type:"number"}
# @markdown >iterations excludes iterations spent during the mse phase, if it is being used. The total iterations will be more if `mse_decay_rate` is more than 0.
iterations = 2000  # @param {type:"number"}
transparent_png = False  # @param {type:"boolean"}

# @markdown <font size="+3">⚠</font> **ADVANCED SETTINGS** <font size="+3">⚠</font>
# @markdown ---
# @markdown ---

# @markdown >If you want to make multiple images with different prompts, use this. Seperate different prompts for different images with a `~` (example: `prompt1~prompt1~prompt3`). Iter is the iterations you want each image to run for. If you use MSE, I'd type a pretty low number (about 10).
multiple_prompt_batches = False  # @param {type:"boolean"}
multiple_prompt_batches_iter = 300  # @param {type:"number"}

# @markdown >`folder_name` is the name of the folder you want to output your result(s) to. Previous outputs will NOT be overwritten. By default, it will be saved to the colab's root folder, but the `save_to_drive` checkbox will save it to `MyDrive\VQGAN_Output` instead.
folder_name = get_path("out")  # @param {type:"string"}
save_to_drive = False  # @param {type:"boolean"}
prompt_experiment = "None"  # @param ['None', 'Fever Dream', 'Philipuss’s Basement', 'Vivid Turmoil', 'Mad Dad', 'Platinum', 'Negative Energy']
if prompt_experiment == "Fever Dream":
    prompts = "<|startoftext|>" + prompts + "<|endoftext|>"
elif prompt_experiment == "Vivid Turmoil":
    prompts = prompts.replace(" ", "¡")
    prompts = "¬" + prompts + "®"
elif prompt_experiment == "Mad Dad":
    prompts = prompts.replace(" ", '\\s+')
elif prompt_experiment == "Platinum":
    prompts = "~!" + prompts + "!~"
    prompts = prompts.replace(" ", '</w>')
elif prompt_experiment == "Philipuss’s Basement":
    prompts = "<|startoftext|>" + prompts
    prompts = prompts.replace(" ", "<|endoftext|><|startoftext|>")
elif prompt_experiment == "Lowercase":
    prompts = prompts.lower()

clip_model = "ViT-B/32"  # @param ["ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
clip_model2 = 'None'  # @param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
clip_model3 = 'None'  # @param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
clip_model4 = 'None'  # @param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
clip_model5 = 'None'  # @param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]
clip_model6 = 'None'  # @param ["None", "ViT-L/14", "ViT-B/32", "ViT-B/16", "RN50x64", "RN50x16", "RN50x4", "RN101", "RN50"]

if clip_model2 == "None": clip_model2 = None
if clip_model3 == "None": clip_model3 = None
if clip_model4 == "None": clip_model4 = None
if clip_model5 == "None": clip_model5 = None
if clip_model6 == "None": clip_model6 = None

# @markdown >Target images work like prompts, write the name of the image. You can add multiple target images by seperating them with a `|`.
target_images = ""  # @param {type:"string"}

# @markdown ><font size="+2">☢</font> Advanced values. Values of cut_pow below 1 prioritize structure over detail, and vice versa for above 1. Step_size affects how wild the change between iterations is, and if final_step_size is not 0, step_size will interpolate towards it over time.
# @markdown >Cutn affects on 'Creativity': less cutout will lead to more random/creative results, sometimes barely readable, while higher values (90+) lead to very stable, photo-like outputs
cutn = 130  # @param {type:"number"}
cut_pow = 1  # @param {type:"number"}
# @markdown >Step_size is like weirdness. Lower: more accurate/realistic, slower; Higher: less accurate/more funky, faster.
step_size = 0.1  # @param {type:"number"}
# @markdown >Start_step_size is a temporary step_size that will be active only in the first 10 iterations. It (sometimes) helps with speed. If it's set to 0, it won't be used.
start_step_size = 0  # @param {type:"number"}
# @markdown >Final_step_size is a goal step_size which the AI will try and reach. If set to 0, it won't be used.
final_step_size = 0  # @param {type:"number"}
if start_step_size <= 0: start_step_size = step_size
if final_step_size <= 0: final_step_size = step_size

# @markdown ---

# @markdown >EMA maintains a moving average of trained parameters. The number below is the rate of decay (higher means slower).
ema_val = 0.98  # @param {type:"number"}

# @markdown >If you want to keep starting from the same point, set `gen_seed` to a positive number. `-1` will make it random every time.
gen_seed = -1  # @param {type:'number'}

images_interval = 10  # @param {type:"number"}

# I think you should give "Free Thoughts on the Proceedings of the Continental Congress" a read, really funny and actually well-written, Hamilton presented it in a bad light IMO.

batch_size = 1  # @param {type:"number"}

# @markdown ---

# @markdown <font size="+1">🔮</font> **MSE Regulization** <font size="+1">🔮</font>
# Based off of this notebook: https://colab.research.google.com/drive/1gFn9u3oPOgsNzJWEFmdK-N9h_y65b8fj?usp=sharing - already in credits
use_mse = True  # @param {type:"boolean"}
mse_images_interval = images_interval
mse_init_weight = 0.2  # @param {type:"number"}
mse_decay_rate = 160  # @param {type:"number"}
mse_epoches = 10  # @param {type:"number"}
##@param {type:"number"}

# @markdown >Overwrites the usual values during the mse phase if included. If any value is 0, its normal counterpart is used instead.
mse_with_zeros = True  # @param {type:"boolean"}
mse_step_size = 0.87  # @param {type:"number"}
mse_cutn = 42  # @param {type:"number"}
mse_cut_pow = 0.75  # @param {type:"number"}

# @markdown >normal_flip_optim flips between two optimizers during the normal (not MSE) phase. It can improve quality, but it's kind of experimental, use at your own risk.
normal_flip_optim = True  # @param {type:"boolean"}
##@markdown >Adding some TV may make the image blurrier but also helps to get rid of noise. A good value to try might be 0.1.
# tv_weight = 0.1 #@param {type:'number'}
# @markdown ---

# @markdown >`altprompts` is a set of prompts that take in a different augmentation pipeline, and can have their own cut_pow. At the moment, the default "alt augment" settings flip the picture cutouts upside down before evaluating. This can be good for optical illusion images. If either cut_pow value is 0, it will use the same value as the normal prompts.
altprompts = ""  # @param {type:"string"}
altprompt_mode = "flipped"
##@param ["normal" , "flipped", "sideways"]
alt_cut_pow = 0  # @param {type:"number"}
alt_mse_cut_pow = 0  # @param {type:"number"}
# altprompt_type = "upside-down" #@param ['upside-down', 'as']

##@markdown ---
##@markdown <font size="+1">💫</font> **Zooming and Moving** <font size="+1">💫</font>
zoom = False
##@param {type:"boolean"}
zoom_speed = 100
##@param {type:"number"}
zoom_frequency = 20
##@param {type:"number"}

# @markdown ---
# @markdown On an unrelated note, if you get any errors while running this, restart the runtime and run the first cell again. If that doesn't work either, message me on Discord (Philipuss#4066).

model_names = {'ImageNet 16384': 'vqgan_imagenet_f16_16384', 'ImageNet 1024': 'vqgan_imagenet_f16_1024',
               "Gumbel 8192": "gumbel_8192", "Sber Gumbel": "sber_gumbel", 'imagenet_cin': 'imagenet_cin',
               'WikiArt 1024': 'wikiart_1024', 'WikiArt 16384': 'wikiart_16384', 'COCO-Stuff': 'coco',
               'FacesHQ': 'faceshq', 'S-FLCKR': 'sflckr', 'WikiArt 7mil': 'wikiart_7mil', 'COCO 1 Stage': 'coco_1stage'}

if template == "Better - Fast":
    prompts = prompts + ". Detailed artwork. ArtStationHQ. unreal engine. 4K HD."
elif template == "Better - Slow":
    prompts = prompts + ". Detailed artwork. Trending on ArtStation. unreal engine. | Rendered in Maya. " + prompts + ". 4K HD."
elif template == "Movie Poster":
    prompts = prompts + ". Movie poster. Rendered in unreal engine. ArtStationHQ."
    width = 400
    height = 592
elif template == 'flag':
    prompts = "A photo of a flag of the country " + prompts + " | Flag of " + prompts + ". White background."
    # import cv2
    # img = cv2.imread('templates/flag.png', 0)
    # cv2.imwrite('templates/final_flag.png', add_noise(img))
    init_image = "templates/flag.png"
    transparent_png = True
elif template == 'planet':
    img = cv2.imread(get_path('templates/planet.png'), 0)
    cv2.imwrite(get_path('templates/final_planet.png'), add_noise(img))
    prompts = "A photo of the planet " + prompts + ". Planet in the middle with black background. | The planet of " + prompts + ". Photo of a planet. Black background. Trending on ArtStation. | Colorful."
    init_image = get_path("templates/final_planet.png")
elif template == 'creature':
    # import cv2
    # img = cv2.imread('templates/planet.png', 0)
    # cv2.imwrite('templates/final_planet.png', add_noise(img))
    prompts = "A photo of a creature with " + prompts + ". Animal in the middle with white background. | The creature has " + prompts + ". Photo of a creature/animal. White background. Detailed image of a creature. | White background."
    init_image = get_path("templates/creature.png")
    # transparent_png = True
elif template == 'Detailed':
    prompts = prompts + ", by Puer Udger. Detailed artwork, trending on artstation. 4K HD, realism."
    flavor = "cumin"
elif template == "human":
    init_image = get_path("templates/human.png")
elif template == "Realistic":
    cutn = 200
    step_size = 0.03
    cut_pow = 0.2
    flavor = "holywater"
elif template == "Consistent Creativity":
    flavor = "cumin"
    cut_pow = 0.01
    cutn = 136
    step_size = 0.08
    mse_step_size = 0.41
    mse_cut_pow = 0.3
    ema_val = 0.99
    normal_flip_optim = False
elif template == "Smooth":
    flavor = "wyvern"
    step_size = 0.10
    cutn = 120
    normal_flip_optim = False
    tv_weight = 10
elif template == "Subtle MSE":
    mse_init_weight = 0.07
    mse_decay_rate = 130
    mse_step_size = 0.2
    mse_cutn = 100
    mse_cut_pow = 0.6
elif template == "Balanced":
    cutn = 130
    cut_pow = 1
    step_size = 0.16
    final_step_size = 0
    ema_val = 0.98
    mse_init_weight = 0.2
    mse_decay_rate = 130
    mse_with_zeros = True
    mse_step_size = 0.9
    mse_cutn = 50
    mse_cut_pow = 0.8
    normal_flip_optim = True
elif template == "Size: Square":
    width = 450
    height = 450
elif template == "Size: Landscape":
    width = 480
    height = 336
elif template == "Size: Poster":
    width = 336
    height = 480
elif template == "Negative Prompt":
    prompts = prompts.replace(":", ":-")
    prompts = prompts.replace(":--", ":")
elif template == "Hyper Fast Results":
    step_size = 1
    ema_val = 0.3
    cutn = 30
elif template == "Better Quality":
    prompts = prompts + ":1 | Watermark, blurry, cropped, confusing, cut, incoherent:-1"

mse_decay = 0

if use_mse == False:
    mse_init_weight = 0.
else:
    mse_decay = mse_init_weight / mse_epoches

if seed == -1:
    seed = None
if init_image == "None":
    init_image = None
if target_images == "None" or not target_images:
    target_images = []
else:
    target_images = target_images.split("|")
    target_images = [image.strip() for image in target_images]

prompts = [phrase.strip() for phrase in prompts.split("|")]
if prompts == ['']:
    prompts = []

altprompts = [phrase.strip() for phrase in altprompts.split("|")]
if altprompts == ['']:
    altprompts = []

if mse_images_interval == 0: mse_images_interval = images_interval
if mse_step_size == 0: mse_step_size = step_size
if mse_cutn == 0: mse_cutn = cutn
if mse_cut_pow == 0: mse_cut_pow = cut_pow
if alt_cut_pow == 0: alt_cut_pow = cut_pow
if alt_mse_cut_pow == 0: alt_mse_cut_pow = mse_cut_pow

augs = nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomSharpness(0.3, p=0.4),
    K.RandomGaussianBlur((3, 3), (4.5, 4.5), p=0.3),
    # K.RandomGaussianNoise(p=0.5),
    # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
    K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),  # padding_mode=2
    K.RandomPerspective(0.2, p=0.4, ),
    K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
    K.RandomGrayscale(p=0.1),
)

if altprompt_mode == "normal":
    altaugs = nn.Sequential(
        K.RandomRotation(degrees=90.0, return_transform=True),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomSharpness(0.3, p=0.4),
        K.RandomGaussianBlur((3, 3), (4.5, 4.5), p=0.3),
        # K.RandomGaussianNoise(p=0.5),
        # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
        K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),  # padding_mode=2
        K.RandomPerspective(0.2, p=0.4, ),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomGrayscale(p=0.1), )
elif altprompt_mode == "flipped":
    altaugs = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        # K.RandomRotation(degrees=90.0),
        K.RandomVerticalFlip(p=1),
        K.RandomSharpness(0.3, p=0.4),
        K.RandomGaussianBlur((3, 3), (4.5, 4.5), p=0.3),
        # K.RandomGaussianNoise(p=0.5),
        # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
        K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),  # padding_mode=2
        K.RandomPerspective(0.2, p=0.4, ),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomGrayscale(p=0.1), )
elif altprompt_mode == "sideways":
    altaugs = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        # K.RandomRotation(degrees=90.0),
        K.RandomVerticalFlip(p=1),
        K.RandomSharpness(0.3, p=0.4),
        K.RandomGaussianBlur((3, 3), (4.5, 4.5), p=0.3),
        # K.RandomGaussianNoise(p=0.5),
        # K.RandomElasticTransform(kernel_size=(33, 33), sigma=(7,7), p=0.2),
        K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode='border'),  # padding_mode=2
        K.RandomPerspective(0.2, p=0.4, ),
        K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
        K.RandomGrayscale(p=0.1), )

if multiple_prompt_batches:
    prompts_all = str(prompts).split("~")
else:
    prompts_all = prompts
    multiple_prompt_batches_iter = iterations

if multiple_prompt_batches:
    mtpl_prmpts_btchs = len(prompts_all)
else:
    mtpl_prmpts_btchs = 1

print(mtpl_prmpts_btchs)

steps_path = get_path('out/steps')
zoom_path = get_path('out/zoom_steps', clear=True)

path = zoom_path

iterations = multiple_prompt_batches_iter

for pr in range(0, mtpl_prmpts_btchs):
    print(prompts_all[pr].replace('[\'', '').replace('\']', ''))
    if multiple_prompt_batches:
        prompts = prompts_all[pr].replace('[\'', '').replace('\']', '')

    if zoom:
        mdf_iter = round(iterations / zoom_frequency)
    else:
        mdf_iter = 2
        zoom_frequency = iterations

    for iter in range(1, mdf_iter):
        if zoom:
            if iter != 0:
                image = Image.open('progress.png')
                area = (0, 0, width - zoom_speed, height - zoom_speed)
                cropped_img = image.crop(area)
                cropped_img.show()

                new_image = cropped_img.resize((width, height))
                new_image.save('zoom.png')
                init_image = 'zoom.png'

        args = Namespace(
            prompts=prompts,
            altprompts=altprompts,
            image_prompts=target_images,
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[width, height],
            init_image=init_image,
            png=transparent_png,
            init_weight=mse_init_weight,
            vqgan_model=model_names[model],
            step_size=step_size,
            start_step_size=start_step_size,
            final_step_size=final_step_size,
            cutn=cutn,
            cut_pow=cut_pow,
            mse_cutn=mse_cutn,
            mse_cut_pow=mse_cut_pow,
            mse_step_size=mse_step_size,
            display_freq=images_interval,
            mse_display_freq=mse_images_interval,
            max_iterations=zoom_frequency,
            mse_end=0,
            flavor=flavor,
            seed=seed,
            folder_name=folder_name,
            save_to_drive=save_to_drive,
            mse_decay_rate=mse_decay_rate,
            mse_decay=mse_decay,
            mse_with_zeros=mse_with_zeros,
            normal_flip_optim=normal_flip_optim,
            ema_val=ema_val,
            augs=augs,
            altaugs=altaugs,
            alt_cut_pow=alt_cut_pow,
            alt_mse_cut_pow=alt_mse_cut_pow,
            is_gumbel=is_gumbel,
            clip_model=clip_model,
            clip_model2=clip_model2,
            clip_model3=clip_model3,
            clip_model4=clip_model4,
            clip_model5=clip_model5,
            clip_model6=clip_model6,
            gen_seed=gen_seed)
        models.args = args

        mh = ModelHost(args)
        x = 0

        for x in range(batch_size):
            mh.setup_model(x)
            last_iter = mh.run(x)
            x = x + 1

        if batch_size != 1:
            # clear_output()
            print("===============================================================================")
            q = 0
            while q < batch_size:
                # display(Image(folder_name + "/" + str(q) + '.png'))
                print("Image" + str(q) + '.png')
                q += 1

    if zoom:
        files = os.listdir(steps_path)
        for index, file in enumerate(files):
            os.rename(os.path.join(steps_path, file),
                      os.path.join(steps_path, ''.join([str(index + 1 + zoom_frequency * iter), '.png'])))
            index = index + 1

        import shutil

        src_path = steps_path
        trg_path = zoom_path

        for src_file in range(1, mdf_iter):
            shutil.move(os.path.join(src_path, src_file), trg_path)
