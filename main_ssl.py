import os
import re
import cv2
import json
import h5py
import time
import math
import subprocess
import wandb
import argparse
import random
import numpy as np
import scipy.stats as stats
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops

import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import torch.nn.utils.prune as prune
from torchvision.datasets import ImageFolder

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.filters as KF

import datasets.loader
import models.evaluate, models.loader, models.loss
import evd.development

torch.backends.cudnn.benchmark = True

##############################
## Hyperparameters
##############################
def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameters for blurring project')

    parser.add_argument('--model_name', type=str, default='resnet50')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=20) 
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='none', help='Learning rate scheduler to use')
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    
    #* self_supervised
    parser.add_argument('--self_supervised', type=int, default=0)

    # time_order
    parser.add_argument('--time_order', type=str, default= '')
    
    # age that get contrast sensitivity 50% of adult 
    parser.add_argument('--cs_age50', type=float, default=4.8*12) #* 4.8*12 for 4.8 years
    parser.add_argument('--mean_freq_factor', type=float, default=1.0) #* 1.0 for acutal mean freq
    parser.add_argument('--std_freq_factor', type=float, default=1.0) #* 1.0 for acutal std freq

    parser.add_argument('--contrast_drop_speed_factor', type=float, default=1.0) #*
    parser.add_argument('--contrast_spd_beta', type=float, default=1.0) #* contrast_spd_beta for contrast increasing speed control
    parser.add_argument('--contrast_thres_map_mode', type=str, default= ['linear_no_beta'][0]) #* 
    parser.add_argument('--decrease_contrast_drop_speed_factor_every_n_month', type=float, default=1) #*
    parser.add_argument('--decrease_speed_of_contrast_drop_speed_factor_every_n_month', type=float, default=0) #* Default 0 means no decrease

    parser.add_argument('--color_development_strategy', type=int, default=0) #*
    parser.add_argument('--color_development_steepness', type=float, default=5) #* # color_development_steepness 
    parser.add_argument('--color_develop_start_epoch', type=int, default=0) #*
    parser.add_argument('--color_develop_lasting_epochs', type=int, default=20) #* # color_develop_lasting_epochs
    # color_speed_factor
    parser.add_argument('--color_speed_factor', type=float, default=1) #* # color_development speed control

    # parser.add_argument('--early_gray_scale_flag', type=int, default=0) #* previous default is using blurring_strage
    parser.add_argument('--blurring_strategy', type=str, default= ['sharp','first_few_epochs_exponentially_decreasing'][0])
    
    parser.add_argument('--blurring_start_epoch', type=int, default=0) #*
    parser.add_argument('--blurring_lasting_epochs', type=int, default=20) #*

    parser.add_argument('--blur_norm_order', type=str, default= ['blur_first','norm_first'][0])

    # development_strategy, months_per_epoch
    parser.add_argument('--development_strategy', type=str, default= 'adult')
    parser.add_argument('--months_per_epoch', type=float, default=1.0)

    parser.add_argument('--byte_flag', type=str, default= ['no_byte','use_byte',                      
                                                            ][0])
    parser.add_argument('--byte_start_epoch', type=int, default=0)
    parser.add_argument('--byte_lasting_epochs', type=int, default=1e3) # IF not remove or assign specific, just inf
    parser.add_argument('--byte_steepness', type=float, default=2)
    parser.add_argument('--using_mixed_byte', type=int, default=0)
    parser.add_argument('--epoch_to_remove_byte', type=int, default=1e3) 
    parser.add_argument('--end_epoch', type=int, default=float('inf')) # IF not remove or assign specific, just inf

    #* apply  blur, color, contrast
    parser.add_argument('--apply_blur', type=int, default=0, help='Flag to apply blur to images')
    parser.add_argument('--apply_color', type=int, default=0, help='Flag to apply color changes')
    parser.add_argument('--apply_contrast', type=int, default=0, help='Flag to apply contrast adjustments')

    parser.add_argument('--dataset', type=str, default= ['ecoset_square256'][0]) # 'texture2shape_miniecoset',
    parser.add_argument('--class_weights_json_path', type=str, default= None) #'/share/klab/datasets/optimized_datasets/lookup_ecoset_json.json')
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--uniform_blur', type=int, default=0) # blur first 20 batch for random order
    parser.add_argument('--mixed_chance_blur', type=int, default=0) # 50% chance to blur
    parser.add_argument('--mixed_chance_blur_start_epoch', type=int, default=0) # default chance start to 0

    parser.add_argument('--grayscale_flag', type=int, default=0) 
    parser.add_argument('--using_mixed_grayscale', type=float, default=0)
    # extra_grayscale_end_months on top of development
    parser.add_argument('--extra_grayscale_end_months', type=float, default=0)


    # Contrast interval mode & Light Intnesity Sensitivity
    parser.add_argument('--contrast_interval_mode', type=str, default='fixed')
    parser.add_argument('--starting_mode', type=str, default='linear')

    parser.add_argument('--normalize_lscs', type=int, default=0)
    parser.add_argument('--multi_stages_lscs', type=int, default=0)

    # csf_leaky_relu_slope
    parser.add_argument('--csf_leaky_relu_slope', type=float, default=0.01)


    parser.add_argument('--show_progress_bar', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_size_val_test', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=150)

    
    return parser.parse_args()

def get_hyp(args):
    """Return hyperparameters as a dictionary."""

    return {
        'dataset': {
            'name': args.dataset,
            'image_size': args.image_size,
            'dataset_path': '/share/klab/datasets/', #'/home/hpczeji1/hpc-work/Codebase/Datasets/',
            'class_weights_json_path': args.class_weights_json_path,
            'augment': { 'randomrotation', 'randomflip', 'grayscale', 'globalcontrast', 'blur', 'globalbrightness','equalize', 'posterize', 'perspective'} if args.using_mixed_grayscale > 0 else  { 'randomrotation', 'randomflip', 'globalcontrast', 'blur', 'globalbrightness', 'equalize', 'posterize', 'perspective'}, # 'randomcrop', #   ,  'grayscale'ormalise happens in the blurring class for training, e.g. 'grayscale',
            'val_test_augment': ['normalize'],
            # 'num_classes': [112,565, 565, 16,1000,118][['texture2shape_miniecoset','ecoset_square256','ecoset_square256_patches','imagenet16','imagenet','facescrub'].index(args.dataset)],
            'grayscale_flag': args.grayscale_flag,
        },
        'network': {
            'model': args.model_name,
            'self_supervised': args.self_supervised,
            'identifier': f'id_{args.id}_lr_{args.learning_rate}',

            'time_order': args.time_order,

            # development_strategy
            'development_strategy': args.development_strategy,
            'months_per_epoch': args.months_per_epoch,

            'blurring_strategy': args.blurring_strategy,
            # 'blurring_decay_steepness': args.blurring_decay_steepness,
            'blurring_start_epoch': args.blurring_start_epoch,
            'blurring_lasting_epochs': args.blurring_lasting_epochs,
            'uniform_blur': args.uniform_blur,
            'mixed_chance_blur': args.mixed_chance_blur,
            'mixed_chance_blur_start_epoch': args.mixed_chance_blur_start_epoch,
            'eval_in_blur': False,

            'color_development_strategy': args.color_development_strategy,
            'color_development_steepness': args.color_development_steepness,
            'color_develop_start_epoch': args.color_develop_start_epoch, 
            'color_develop_lasting_epochs': args.color_develop_lasting_epochs,
            'using_mixed_grayscale': args.using_mixed_grayscale,
            # 'early_gray_scale_flag': args.early_gray_scale_flag,
            'color_speed_factor': args.color_speed_factor,
            'extra_grayscale_end_months': args.extra_grayscale_end_months,
            
            'byte_flag': args.byte_flag,
            'contrast_interval_mode': args.contrast_interval_mode,
            'starting_mode': args.starting_mode,
            'contrast_drop_speed_factor': args.contrast_drop_speed_factor,
            'decrease_contrast_drop_speed_factor_every_n_month': args.decrease_contrast_drop_speed_factor_every_n_month,
            'decrease_speed_of_contrast_drop_speed_factor_every_n_month': args.decrease_speed_of_contrast_drop_speed_factor_every_n_month,
            'contrast_thres_map_mode': args.contrast_thres_map_mode,
            'contrast_spd_beta': args.contrast_spd_beta,

            # cs_age50
            'cs_age50': args.cs_age50,
            'mean_freq_factor': args.mean_freq_factor,
            'std_freq_factor': args.std_freq_factor,
            'csf_leaky_relu_slope': args.csf_leaky_relu_slope, #* soft threhold for contrast 

            #* apply blur, color, contrast
            'apply_blur': args.apply_blur,
            'apply_color': args.apply_color,
            'apply_contrast': args.apply_contrast,

            'byte_start_epoch': args.byte_start_epoch,
            'byte_lasting_epochs': args.byte_lasting_epochs,
            'byte_steepness': args.byte_steepness,
            'using_mixed_byte': args.using_mixed_byte,
            'epoch_to_remove_byte': args.epoch_to_remove_byte,
            
            'blur_norm_order': args.blur_norm_order,
            'pretrained': args.pretrained,
            'end_epoch': args.end_epoch,

            'normalize_lscs': args.normalize_lscs,
            'multi_stages_lscs': args.multi_stages_lscs,
        },
        'optimizer': {
            'type': 'adam',
            'lr': args.learning_rate,
            'lr_scheduler': args.lr_scheduler,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'device': 'cuda',
            'dataloader': {
                'num_workers_train': 10, # number of cpu workers processing the batches 
                'prefetch_factor_train': 4, # number of batches kept in memory by each worker (providing quick access for the gpu)
                'num_workers_val_test': 3, # do not need lots of workers for val/test
                'prefetch_factor_val_test': 4 
                },
            # 'dataloader': {
            #     'num_workers_train': 10, #30, #10, #* stuck in N*num_workers_train epochs
            #     'prefetch_factor_train': 1, # 4
            #     'num_workers_val_test': 3, # 10, # 
            #     'prefetch_factor_val_test': 1, # 4
            # },
            'show_progress_bar': args.show_progress_bar,
            'seed': args.seed,
            'step_size': args.step_size,
            'gamma': args.gamma,
        },
        'misc': {
            'use_amp': True,
            'batch_size_val_test': args.batch_size_val_test,
            'save_logs': 5,
            'save_net': 5,
            'project_name': "Dec_4thYear_texture2shape_project", #* Saving name for wandb logging

        },
        "categories": {
            'imagenet16': ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock', 'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck', 'car', 'bird', 'dog'],
            'imagenet':  ["b'tench'", "b'goldfish'", "b'great_white_shark'", "b'tiger_shark'", "b'hammerhead'", "b'electric_ray'", "b'stingray'", "b'cock'", "b'hen'", "b'ostrich'", "b'brambling'", "b'goldfinch'", "b'house_finch'", "b'junco'", "b'indigo_bunting'", "b'robin'", "b'bulbul'", "b'jay'", "b'magpie'", "b'chickadee'", "b'water_ouzel'", "b'kite'", "b'bald_eagle'", "b'vulture'", "b'great_grey_owl'", "b'European_fire_salamander'", "b'common_newt'", "b'eft'", "b'spotted_salamander'", "b'axolotl'", "b'bullfrog'", "b'tree_frog'", "b'tailed_frog'", "b'loggerhead'", "b'leatherback_turtle'", "b'mud_turtle'", "b'terrapin'", "b'box_turtle'", "b'banded_gecko'", "b'common_iguana'", "b'American_chameleon'", "b'whiptail'", "b'agama'", "b'frilled_lizard'", "b'alligator_lizard'", "b'Gila_monster'", "b'green_lizard'", "b'African_chameleon'", "b'Komodo_dragon'", "b'African_crocodile'", "b'American_alligator'", "b'triceratops'", "b'thunder_snake'", "b'ringneck_snake'", "b'hognose_snake'", "b'green_snake'", "b'king_snake'", "b'garter_snake'", "b'water_snake'", "b'vine_snake'", "b'night_snake'", "b'boa_constrictor'", "b'rock_python'", "b'Indian_cobra'", "b'green_mamba'", "b'sea_snake'", "b'horned_viper'", "b'diamondback'", "b'sidewinder'", "b'trilobite'", "b'harvestman'", "b'scorpion'", "b'black_and_gold_garden_spider'", "b'barn_spider'", "b'garden_spider'", "b'black_widow'", "b'tarantula'", "b'wolf_spider'", "b'tick'", "b'centipede'", "b'black_grouse'", "b'ptarmigan'", "b'ruffed_grouse'", "b'prairie_chicken'", "b'peacock'", "b'quail'", "b'partridge'", "b'African_grey'", "b'macaw'", "b'sulphur-crested_cockatoo'", "b'lorikeet'", "b'coucal'", "b'bee_eater'", "b'hornbill'", "b'hummingbird'", "b'jacamar'", "b'toucan'", "b'drake'", "b'red-breasted_merganser'", "b'goose'", "b'black_swan'", "b'tusker'", "b'echidna'", "b'platypus'", "b'wallaby'", "b'koala'", "b'wombat'", "b'jellyfish'", "b'sea_anemone'", "b'brain_coral'", "b'flatworm'", "b'nematode'", "b'conch'", "b'snail'", "b'slug'", "b'sea_slug'", "b'chiton'", "b'chambered_nautilus'", "b'Dungeness_crab'", "b'rock_crab'", "b'fiddler_crab'", "b'king_crab'", "b'American_lobster'", "b'spiny_lobster'", "b'crayfish'", "b'hermit_crab'", "b'isopod'", "b'white_stork'", "b'black_stork'", "b'spoonbill'", "b'flamingo'", "b'little_blue_heron'", "b'American_egret'", "b'bittern'", "b'crane'", "b'limpkin'", "b'European_gallinule'", "b'American_coot'", "b'bustard'", "b'ruddy_turnstone'", "b'red-backed_sandpiper'", "b'redshank'", "b'dowitcher'", "b'oystercatcher'", "b'pelican'", "b'king_penguin'", "b'albatross'", "b'grey_whale'", "b'killer_whale'", "b'dugong'", "b'sea_lion'", "b'Chihuahua'", "b'Japanese_spaniel'", "b'Maltese_dog'", "b'Pekinese'", "b'Shih-Tzu'", "b'Blenheim_spaniel'", "b'papillon'", "b'toy_terrier'", "b'Rhodesian_ridgeback'", "b'Afghan_hound'", "b'basset'", "b'beagle'", "b'bloodhound'", "b'bluetick'", "b'black-and-tan_coonhound'", "b'Walker_hound'", "b'English_foxhound'", "b'redbone'", "b'borzoi'", "b'Irish_wolfhound'", "b'Italian_greyhound'", "b'whippet'", "b'Ibizan_hound'", "b'Norwegian_elkhound'", "b'otterhound'", "b'Saluki'", "b'Scottish_deerhound'", "b'Weimaraner'", "b'Staffordshire_bullterrier'", "b'American_Staffordshire_terrier'", "b'Bedlington_terrier'", "b'Border_terrier'", "b'Kerry_blue_terrier'", "b'Irish_terrier'", "b'Norfolk_terrier'", "b'Norwich_terrier'", "b'Yorkshire_terrier'", "b'wire-haired_fox_terrier'", "b'Lakeland_terrier'", "b'Sealyham_terrier'", "b'Airedale'", "b'cairn'", "b'Australian_terrier'", "b'Dandie_Dinmont'", "b'Boston_bull'", "b'miniature_schnauzer'", "b'giant_schnauzer'", "b'standard_schnauzer'", "b'Scotch_terrier'", "b'Tibetan_terrier'", "b'silky_terrier'", "b'soft-coated_wheaten_terrier'", "b'West_Highland_white_terrier'", "b'Lhasa'", "b'flat-coated_retriever'", "b'curly-coated_retriever'", "b'golden_retriever'", "b'Labrador_retriever'", "b'Chesapeake_Bay_retriever'", "b'German_short-haired_pointer'", "b'vizsla'", "b'English_setter'", "b'Irish_setter'", "b'Gordon_setter'", "b'Brittany_spaniel'", "b'clumber'", "b'English_springer'", "b'Welsh_springer_spaniel'", "b'cocker_spaniel'", "b'Sussex_spaniel'", "b'Irish_water_spaniel'", "b'kuvasz'", "b'schipperke'", "b'groenendael'", "b'malinois'", "b'briard'", "b'kelpie'", "b'komondor'", "b'Old_English_sheepdog'", "b'Shetland_sheepdog'", "b'collie'", "b'Border_collie'", "b'Bouvier_des_Flandres'", "b'Rottweiler'", "b'German_shepherd'", "b'Doberman'", "b'miniature_pinscher'", "b'Greater_Swiss_Mountain_dog'", "b'Bernese_mountain_dog'", "b'Appenzeller'", "b'EntleBucher'", "b'boxer'", "b'bull_mastiff'", "b'Tibetan_mastiff'", "b'French_bulldog'", "b'Great_Dane'", "b'Saint_Bernard'", "b'Eskimo_dog'", "b'malamute'", "b'Siberian_husky'", "b'dalmatian'", "b'affenpinscher'", "b'basenji'", "b'pug'", "b'Leonberg'", "b'Newfoundland'", "b'Great_Pyrenees'", "b'Samoyed'", "b'Pomeranian'", "b'chow'", "b'keeshond'", "b'Brabancon_griffon'", "b'Pembroke'", "b'Cardigan'", "b'toy_poodle'", "b'miniature_poodle'", "b'standard_poodle'", "b'Mexican_hairless'", "b'timber_wolf'", "b'white_wolf'", "b'red_wolf'", "b'coyote'", "b'dingo'", "b'dhole'", "b'African_hunting_dog'", "b'hyena'", "b'red_fox'", "b'kit_fox'", "b'Arctic_fox'", "b'grey_fox'", "b'tabby'", "b'tiger_cat'", "b'Persian_cat'", "b'Siamese_cat'", "b'Egyptian_cat'", "b'cougar'", "b'lynx'", "b'leopard'", "b'snow_leopard'", "b'jaguar'", "b'lion'", "b'tiger'", "b'cheetah'", "b'brown_bear'", "b'American_black_bear'", "b'ice_bear'", "b'sloth_bear'", "b'mongoose'", "b'meerkat'", "b'tiger_beetle'", "b'ladybug'", "b'ground_beetle'", "b'long-horned_beetle'", "b'leaf_beetle'", "b'dung_beetle'", "b'rhinoceros_beetle'", "b'weevil'", "b'fly'", "b'bee'", "b'ant'", "b'grasshopper'", "b'cricket'", "b'walking_stick'", "b'cockroach'", "b'mantis'", "b'cicada'", "b'leafhopper'", "b'lacewing'", "b'dragonfly'", "b'damselfly'", "b'admiral'", "b'ringlet'", "b'monarch'", "b'cabbage_butterfly'", "b'sulphur_butterfly'", "b'lycaenid'", "b'starfish'", "b'sea_urchin'", "b'sea_cucumber'", "b'wood_rabbit'", "b'hare'", "b'Angora'", "b'hamster'", "b'porcupine'", "b'fox_squirrel'", "b'marmot'", "b'beaver'", "b'guinea_pig'", "b'sorrel'", "b'zebra'", "b'hog'", "b'wild_boar'", "b'warthog'", "b'hippopotamus'", "b'ox'", "b'water_buffalo'", "b'bison'", "b'ram'", "b'bighorn'", "b'ibex'", "b'hartebeest'", "b'impala'", "b'gazelle'", "b'Arabian_camel'", "b'llama'", "b'weasel'", "b'mink'", "b'polecat'", "b'black-footed_ferret'", "b'otter'", "b'skunk'", "b'badger'", "b'armadillo'", "b'three-toed_sloth'", "b'orangutan'", "b'gorilla'", "b'chimpanzee'", "b'gibbon'", "b'siamang'", "b'guenon'", "b'patas'", "b'baboon'", "b'macaque'", "b'langur'", "b'colobus'", "b'proboscis_monkey'", "b'marmoset'", "b'capuchin'", "b'howler_monkey'", "b'titi'", "b'spider_monkey'", "b'squirrel_monkey'", "b'Madagascar_cat'", "b'indri'", "b'Indian_elephant'", "b'African_elephant'", "b'lesser_panda'", "b'giant_panda'", "b'barracouta'", "b'eel'", "b'coho'", "b'rock_beauty'", "b'anemone_fish'", "b'sturgeon'", "b'gar'", "b'lionfish'", "b'puffer'", "b'abacus'", "b'abaya'", "b'academic_gown'", "b'accordion'", "b'acoustic_guitar'", "b'aircraft_carrier'", "b'airliner'", "b'airship'", "b'altar'", "b'ambulance'", "b'amphibian'", "b'analog_clock'", "b'apiary'", "b'apron'", "b'ashcan'", "b'assault_rifle'", "b'backpack'", "b'bakery'", "b'balance_beam'", "b'balloon'", "b'ballpoint'", "b'Band_Aid'", "b'banjo'", "b'bannister'", "b'barbell'", "b'barber_chair'", "b'barbershop'", "b'barn'", "b'barometer'", "b'barrel'", "b'barrow'", "b'baseball'", "b'basketball'", "b'bassinet'", "b'bassoon'", "b'bathing_cap'", "b'bath_towel'", "b'bathtub'", "b'beach_wagon'", "b'beacon'", "b'beaker'", "b'bearskin'", "b'beer_bottle'", "b'beer_glass'", "b'bell_cote'", "b'bib'", "b'bicycle-built-for-two'", "b'bikini'", "b'binder'", "b'binoculars'", "b'birdhouse'", "b'boathouse'", "b'bobsled'", "b'bolo_tie'", "b'bonnet'", "b'bookcase'", "b'bookshop'", "b'bottlecap'", "b'bow'", "b'bow_tie'", "b'brass'", "b'brassiere'", "b'breakwater'", "b'breastplate'", "b'broom'", "b'bucket'", "b'buckle'", "b'bulletproof_vest'", "b'bullet_train'", "b'butcher_shop'", "b'cab'", "b'caldron'", "b'candle'", "b'cannon'", "b'canoe'", "b'can_opener'", "b'cardigan'", "b'car_mirror'", "b'carousel'", "b'carpenter's_kit'", "b'carton'", "b'car_wheel'", "b'cash_machine'", "b'cassette'", "b'cassette_player'", "b'castle'", "b'catamaran'", "b'CD_player'", "b'cello'", "b'cellular_telephone'", "b'chain'", "b'chainlink_fence'", "b'chain_mail'", "b'chain_saw'", "b'chest'", "b'chiffonier'", "b'chime'", "b'china_cabinet'", "b'Christmas_stocking'", "b'church'", "b'cinema'", "b'cleaver'", "b'cliff_dwelling'", "b'cloak'", "b'clog'", "b'cocktail_shaker'", "b'coffee_mug'", "b'coffeepot'", "b'coil'", "b'combination_lock'", "b'computer_keyboard'", "b'confectionery'", "b'container_ship'", "b'convertible'", "b'corkscrew'", "b'cornet'", "b'cowboy_boot'", "b'cowboy_hat'", "b'cradle'", "b'crane'", "b'crash_helmet'", "b'crate'", "b'crib'", "b'Crock_Pot'", "b'croquet_ball'", "b'crutch'", "b'cuirass'", "b'dam'", "b'desk'", "b'desktop_computer'", "b'dial_telephone'", "b'diaper'", "b'digital_clock'", "b'digital_watch'", "b'dining_table'", "b'dishrag'", "b'dishwasher'", "b'disk_brake'", "b'dock'", "b'dogsled'", "b'dome'", "b'doormat'", "b'drilling_platform'", "b'drum'", "b'drumstick'", "b'dumbbell'", "b'Dutch_oven'", "b'electric_fan'", "b'electric_guitar'", "b'electric_locomotive'", "b'entertainment_center'", "b'envelope'", "b'espresso_maker'", "b'face_powder'", "b'feather_boa'", "b'file'", "b'fireboat'", "b'fire_engine'", "b'fire_screen'", "b'flagpole'", "b'flute'", "b'folding_chair'", "b'football_helmet'", "b'forklift'", "b'fountain'", "b'fountain_pen'", "b'four-poster'", "b'freight_car'", "b'French_horn'", "b'frying_pan'", "b'fur_coat'", "b'garbage_truck'", "b'gasmask'", "b'gas_pump'", "b'goblet'", "b'go-kart'", "b'golf_ball'", "b'golfcart'", "b'gondola'", "b'gong'", "b'gown'", "b'grand_piano'", "b'greenhouse'", "b'grille'", "b'grocery_store'", "b'guillotine'", "b'hair_slide'", "b'hair_spray'", "b'half_track'", "b'hammer'", "b'hamper'", "b'hand_blower'", "b'hand-held_computer'", "b'handkerchief'", "b'hard_disc'", "b'harmonica'", "b'harp'", "b'harvester'", "b'hatchet'", "b'holster'", "b'home_theater'", "b'honeycomb'", "b'hook'", "b'hoopskirt'", "b'horizontal_bar'", "b'horse_cart'", "b'hourglass'", "b'iPod'", "b'iron'", "b'jack-o'-lantern'", "b'jean'", "b'jeep'", "b'jersey'", "b'jigsaw_puzzle'", "b'jinrikisha'", "b'joystick'", "b'kimono'", "b'knee_pad'", "b'knot'", "b'lab_coat'", "b'ladle'", "b'lampshade'", "b'laptop'", "b'lawn_mower'", "b'lens_cap'", "b'letter_opener'", "b'library'", "b'lifeboat'", "b'lighter'", "b'limousine'", "b'liner'", "b'lipstick'", "b'Loafer'", "b'lotion'", "b'loudspeaker'", "b'loupe'", "b'lumbermill'", "b'magnetic_compass'", "b'mailbag'", "b'mailbox'", "b'maillot'", "b'maillot'", "b'manhole_cover'", "b'maraca'", "b'marimba'", "b'mask'", "b'matchstick'", "b'maypole'", "b'maze'", "b'measuring_cup'", "b'medicine_chest'", "b'megalith'", "b'microphone'", "b'microwave'", "b'military_uniform'", "b'milk_can'", "b'minibus'", "b'miniskirt'", "b'minivan'", "b'missile'", "b'mitten'", "b'mixing_bowl'", "b'mobile_home'", "b'Model_T'", "b'modem'", "b'monastery'", "b'monitor'", "b'moped'", "b'mortar'", "b'mortarboard'", "b'mosque'", "b'mosquito_net'", "b'motor_scooter'", "b'mountain_bike'", "b'mountain_tent'", "b'mouse'", "b'mousetrap'", "b'moving_van'", "b'muzzle'", "b'nail'", "b'neck_brace'", "b'necklace'", "b'nipple'", "b'notebook'", "b'obelisk'", "b'oboe'", "b'ocarina'", "b'odometer'", "b'oil_filter'", "b'organ'", "b'oscilloscope'", "b'overskirt'", "b'oxcart'", "b'oxygen_mask'", "b'packet'", "b'paddle'", "b'paddlewheel'", "b'padlock'", "b'paintbrush'", "b'pajama'", "b'palace'", "b'panpipe'", "b'paper_towel'", "b'parachute'", "b'parallel_bars'", "b'park_bench'", "b'parking_meter'", "b'passenger_car'", "b'patio'", "b'pay-phone'", "b'pedestal'", "b'pencil_box'", "b'pencil_sharpener'", "b'perfume'", "b'Petri_dish'", "b'photocopier'", "b'pick'", "b'pickelhaube'", "b'picket_fence'", "b'pickup'", "b'pier'", "b'piggy_bank'", "b'pill_bottle'", "b'pillow'", "b'ping-pong_ball'", "b'pinwheel'", "b'pirate'", "b'pitcher'", "b'plane'", "b'planetarium'", "b'plastic_bag'", "b'plate_rack'", "b'plow'", "b'plunger'", "b'Polaroid_camera'", "b'pole'", "b'police_van'", "b'poncho'", "b'pool_table'", "b'pop_bottle'", "b'pot'", "b'potter's_wheel'", "b'power_drill'", "b'prayer_rug'", "b'printer'", "b'prison'", "b'projectile'", "b'projector'", "b'puck'", "b'punching_bag'", "b'purse'", "b'quill'", "b'quilt'", "b'racer'", "b'racket'", "b'radiator'", "b'radio'", "b'radio_telescope'", "b'rain_barrel'", "b'recreational_vehicle'", "b'reel'", "b'reflex_camera'", "b'refrigerator'", "b'remote_control'", "b'restaurant'", "b'revolver'", "b'rifle'", "b'rocking_chair'", "b'rotisserie'", "b'rubber_eraser'", "b'rugby_ball'", "b'rule'", "b'running_shoe'", "b'safe'", "b'safety_pin'", "b'saltshaker'", "b'sandal'", "b'sarong'", "b'sax'", "b'scabbard'", "b'scale'", "b'school_bus'", "b'schooner'", "b'scoreboard'", "b'screen'", "b'screw'", "b'screwdriver'", "b'seat_belt'", "b'sewing_machine'", "b'shield'", "b'shoe_shop'", "b'shoji'", "b'shopping_basket'", "b'shopping_cart'", "b'shovel'", "b'shower_cap'", "b'shower_curtain'", "b'ski'", "b'ski_mask'", "b'sleeping_bag'", "b'slide_rule'", "b'sliding_door'", "b'slot'", "b'snorkel'", "b'snowmobile'", "b'snowplow'", "b'soap_dispenser'", "b'soccer_ball'", "b'sock'", "b'solar_dish'", "b'sombrero'", "b'soup_bowl'", "b'space_bar'", "b'space_heater'", "b'space_shuttle'", "b'spatula'", "b'speedboat'", "b'spider_web'", "b'spindle'", "b'sports_car'", "b'spotlight'", "b'stage'", "b'steam_locomotive'", "b'steel_arch_bridge'", "b'steel_drum'", "b'stethoscope'", "b'stole'", "b'stone_wall'", "b'stopwatch'", "b'stove'", "b'strainer'", "b'streetcar'", "b'stretcher'", "b'studio_couch'", "b'stupa'", "b'submarine'", "b'suit'", "b'sundial'", "b'sunglass'", "b'sunglasses'", "b'sunscreen'", "b'suspension_bridge'", "b'swab'", "b'sweatshirt'", "b'swimming_trunks'", "b'swing'", "b'switch'", "b'syringe'", "b'table_lamp'", "b'tank'", "b'tape_player'", "b'teapot'", "b'teddy'", "b'television'", "b'tennis_ball'", "b'thatch'", "b'theater_curtain'", "b'thimble'", "b'thresher'", "b'throne'", "b'tile_roof'", "b'toaster'", "b'tobacco_shop'", "b'toilet_seat'", "b'torch'", "b'totem_pole'", "b'tow_truck'", "b'toyshop'", "b'tractor'", "b'trailer_truck'", "b'tray'", "b'trench_coat'", "b'tricycle'", "b'trimaran'", "b'tripod'", "b'triumphal_arch'", "b'trolleybus'", "b'trombone'", "b'tub'", "b'turnstile'", "b'typewriter_keyboard'", "b'umbrella'", "b'unicycle'", "b'upright'", "b'vacuum'", "b'vase'", "b'vault'", "b'velvet'", "b'vending_machine'", "b'vestment'", "b'viaduct'", "b'violin'", "b'volleyball'", "b'waffle_iron'", "b'wall_clock'", "b'wallet'", "b'wardrobe'", "b'warplane'", "b'washbasin'", "b'washer'", "b'water_bottle'", "b'water_jug'", "b'water_tower'", "b'whiskey_jug'", "b'whistle'", "b'wig'", "b'window_screen'", "b'window_shade'", "b'Windsor_tie'", "b'wine_bottle'", "b'wing'", "b'wok'", "b'wooden_spoon'", "b'wool'", "b'worm_fence'", "b'wreck'", "b'yawl'", "b'yurt'", "b'web_site'", "b'comic_book'", "b'crossword_puzzle'", "b'street_sign'", "b'traffic_light'", "b'book_jacket'", "b'menu'", "b'plate'", "b'guacamole'", "b'consomme'", "b'hot_pot'", "b'trifle'", "b'ice_cream'", "b'ice_lolly'", "b'French_loaf'", "b'bagel'", "b'pretzel'", "b'cheeseburger'", "b'hotdog'", "b'mashed_potato'", "b'head_cabbage'", "b'broccoli'", "b'cauliflower'", "b'zucchini'", "b'spaghetti_squash'", "b'acorn_squash'", "b'butternut_squash'", "b'cucumber'", "b'artichoke'", "b'bell_pepper'", "b'cardoon'", "b'mushroom'", "b'Granny_Smith'", "b'strawberry'", "b'orange'", "b'lemon'", "b'fig'", "b'pineapple'", "b'banana'", "b'jackfruit'", "b'custard_apple'", "b'pomegranate'", "b'hay'", "b'carbonara'", "b'chocolate_sauce'", "b'dough'", "b'meat_loaf'", "b'pizza'", "b'potpie'", "b'burrito'", "b'red_wine'", "b'espresso'", "b'cup'", "b'eggnog'", "b'alp'", "b'bubble'", "b'cliff'", "b'coral_reef'", "b'geyser'", "b'lakeside'", "b'promontory'", "b'sandbar'", "b'seashore'", "b'valley'", "b'volcano'", "b'ballplayer'", "b'groom'", "b'scuba_diver'", "b'rapeseed'", "b'daisy'", "b'yellow_lady's_slipper'", "b'corn'", "b'acorn'", "b'hip'", "b'buckeye'", "b'coral_fungus'", "b'agaric'", "b'gyromitra'", "b'stinkhorn'", "b'earthstar'", "b'hen-of-the-woods'", "b'bolete'", "b'ear'", "b'toilet_tissue'"],

        },
        # Color Vision: Parameters for each type chromatic sensitivity development to fitting the t function curve
        "params": {
            "Protan (Red)": {"a": 7.902e-3, "b": 2.740e-5, "alpha": 0.928},
            "Deutan (Green)": {"a": 7.372e-3, "b": 2.749e-5, "alpha": 0.920},
            "Tritan (Blue)": {"a": 10.592e-3, "b": 8.089e-5, "alpha": 0.831}
        }
    }



def train_epoch(epoch, net, train_loader, optimizer, criterion, scaler, hyp):
    """
    Train the network for one epoch.

    Args:
        epoch (int): Current epoch number.
        net (torch.nn.Module): Neural network model.
        train_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Loss function.
        scaler (torch.amp.GradScaler): Gradient scaler for mixed precision training.
        hyp (dict): Hyperparameters.

    Returns:
        tuple: Average training loss and accuracy for the epoch.
    """

    
    before_start = time.time()
    net.train()
    device = hyp['optimizer']['device']
    net.to(device).float()
    train_loss, train_acc = 0.0, 0.0

    # Generate age months curve to map batches to age months
    age_months_curve = evd.development.generate_age_months_curve(hyp['optimizer']['n_epochs'], len(train_loader), hyp['network']['months_per_epoch'], 
                                                 mid_phase=hyp['network']['time_order'] == 'mid_phase', shuffle=hyp['network']['time_order'] == 'random', seed=hyp['optimizer']['seed'])

    start = time.time()
    self_supervised = hyp['network'].get('self_supervised', False)
    # print("time start", time.time()-start)
    for batch_id, (imgs, lbls) in enumerate(train_loader):

        if not self_supervised:
            imgs, lbls = imgs.to(device), lbls.to(device)
            imgs = imgs.squeeze(1)  #*  [512, 1, 3, 256, 256] -> [512, 3, 256, 256]
        else:
            # For self-supervised learning, data might be different
            # For example, data could be (img1, img2)
            imgs[0] = imgs[0].to(device).squeeze(1)
            imgs[1] = imgs[1].to(device).squeeze(1)

        optimizer.zero_grad()

        # Apply transformations based on development strategy
        age_months = age_months_curve[(epoch - 1) * len(train_loader) + batch_id]

        # Blur
        apply_blur = hyp['network']['apply_blur']
        # Color
        apply_color = hyp['network']['apply_color']
        color_speed_factor = hyp['network']['color_speed_factor']
        # Contrast
        apply_contrast = hyp['network']['apply_contrast']

        cs_age50=hyp['network']['cs_age50']

        
        # Apply transformations across time in development
        if hyp["network"]["development_strategy"] == 'retina_fft_development':

            decrease_contrast_drop_speed_factor_every_n_month = hyp['network']['decrease_contrast_drop_speed_factor_every_n_month'] 
            decrease_speed_of_contrast_drop_speed_factor_every_n_month = hyp['network']['decrease_speed_of_contrast_drop_speed_factor_every_n_month']
            contrast_control_coeff = math.floor(age_months / decrease_contrast_drop_speed_factor_every_n_month)* decrease_speed_of_contrast_drop_speed_factor_every_n_month #* 0*2, 1*2,
            contrast_control_coeff = max(contrast_control_coeff, 1) # need to larget than 1

            imgs = evd.development.EarlyVisualDevelopmentTransformer().apply_fft_transformations(imgs, age_months, cs_age50, mean_freq_factor=hyp['network']['mean_freq_factor'], std_freq_factor=hyp['network']['std_freq_factor'],  contrast_drop_speed_factor = hyp['network']['contrast_drop_speed_factor']/contrast_control_coeff, image_size=hyp['dataset']['image_size'], verbose=False)
                                                                        # apply_blur, apply_color, apply_contrast, 
                                                                        # use_fft_contrast, use_soft_threshold, use_csf_U_filter, csf_leaky_relu_slope, color_speed_factor, 
                                                                        # contrast_threshold)
        elif hyp["network"]["development_strategy"] == 'adult':
            pass
        else:
            raise ValueError(f"Unknown development strategy: {hyp['network']['development_strategy']}")
        # print("time after fft", time.time()-start)

        # Compute the forward pass and loss.
        if hyp['optimizer']['device'] == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                if self_supervised:
                    # Compute self-supervised loss (e.g., contrastive loss)
                    # import pdb; pdb.set_trace()
                    p1, p2, z1, z2 = net(x1=imgs[0], x2=imgs[1])

                    # Normalize to unit vectors
                    p1 = torch.nn.functional.normalize(p1, dim=1)  # Normalize along the feature dimension
                    p2 = torch.nn.functional.normalize(p2, dim=1)
                    z1 = torch.nn.functional.normalize(z1, dim=1)
                    z2 = torch.nn.functional.normalize(z2, dim=1)
                    loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                else:
                    outputs = net(imgs)
                    loss = criterion(outputs, lbls.long())
        else:
            raise ValueError("Invalid device")
        
        # print("time after loss", time.time()-start)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        train_loss += loss.item()
        if not self_supervised:
            train_acc += models.evaluate.compute_accuracy(outputs, lbls)
 
        if hyp['optimizer']['show_progress_bar']:
            print(f'Training Epoch {epoch}: Batch {batch_id} of {len(train_loader)}', end="\r")
        
        # print("time end", time.time()-start,'\n')

    avg_train_loss = train_loss / len(train_loader)
    if not self_supervised:
        avg_train_acc = train_acc / len(train_loader)
    
    print(f'\nEpoch {epoch} completed in {time.time() - start:.2f} seconds')
    if not self_supervised:
        return avg_train_loss, avg_train_acc
    else:
        return avg_train_loss, 0.0



if __name__ == '__main__':

    # Get the hyperparameters
    args = get_args()
    hyp = get_hyp(args)

    # Ensure reproducibility
    torch.manual_seed(hyp['optimizer']['seed'])
    np.random.seed(hyp['optimizer']['seed'])
    random.seed(hyp['optimizer']['seed'])

    # Load datasets
    train_loader, val_loader, test_loader, hyp = datasets.loader.get_dataset_loaders(hyp, ['train', 'val', 'test'], self_supervised=hyp['network']['self_supervised'])

    # Initialize network and optimizer
    net, net_name = models.loader.get_network_model(hyp)
    optimizer =  models.loader.get_optimizer(hyp, net)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set up the scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=hyp['misc'].get('use_amp', False))

    # Set up the loss function --> differnt loss function for different dataset | class weights
    criterion = models.loss.get_loss_function(hyp, device)

    # Save initial learning rates in optimizer param groups
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
    # Set up the learning rate scheduler
    lr_scheduler =  models.loader.get_lr_scheduler(optimizer, hyp['optimizer'].get('lr_scheduler'), hyp)

    # Initialize Weights & Biases (WandB) for experiment tracking
    wandb.init(
        project=hyp['misc'].get('project_name', "Dec_4thYear_Blurring_project"),
        name=net_name,
        config=hyp
    )

    # Initialize or resume training
    net, logs, start_epoch, log_path, net_path = models.loader.initialize_or_resume_training(net, net_name, hyp)
    net = net.float().to(device)

    # Training loop
    n_epochs = hyp['optimizer']['n_epochs']
    end_epoch = hyp['network'].get('end_epoch', float('inf'))
    self_supervised = hyp['network'].get('self_supervised', False)

    for epoch in range(start_epoch, n_epochs + 1):
        # Early stopping condition
        if epoch > end_epoch:
            break

        # Update the current epoch in hyperparameters
        hyp["epoch"] = epoch

        # Training phase
        train_loss, train_acc = train_epoch(
            epoch, net, train_loader, optimizer, criterion, scaler, hyp
        )

        # Validation phase
        if not self_supervised:
            val_loss, val_acc =  models.evaluate.validate_epoch(net, val_loader, criterion, hyp)
        else:
            val_loss, val_acc = 0.0, 0.0 #* no need since no trainsformation of Simsaime will be a same image

        # Step the learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Log metrics
        logs['train_losses'][epoch] = train_loss
        logs['val_losses'][epoch] = val_loss
        logs['train_accuracies'][epoch] = train_acc
        logs['val_accuracies'][epoch] = val_acc

        # Save metrics and model checkpoints
        models.loader.log_and_save_metrics(
            epoch, logs, net, optimizer, log_path, net_path, net_name, hyp
        )

    print('\nTraining completed! Evaluating on the test set...\n')

    # Save the final model checkpoint if not already saved
    final_model_path = f'{net_path}/{net_name}_epoch_{epoch}.pth'
    if not os.path.exists(final_model_path):
        torch.save(net.state_dict(), final_model_path)

    # Evaluate the model on the test set
    models.evaluate.evaluate_on_test_set(net, test_loader, criterion, hyp)

    print('\nTraining and evaluation completed successfully!\n')








