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
import torchvision
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
from torchvision.datasets import ImageFolder

import torch
import torch.nn as nn
import kornia
import kornia.augmentation as K
import kornia.filters as KF

# Loading the dataset loaders
def get_dataset_loaders(hyp, splits, in_memory =True, compute_stats=False, self_supervised=False):
    """Return train, validation, and test dataloaders based on given hyperparameters."""
    if hyp['dataset']['name'] == 'texture2shape_miniecoset':
        dataset_path = f"{hyp['dataset']['dataset_path']}texture2shape_miniecoset_{hyp['dataset']['image_size']}px.h5"
        print(f"Loading dataset from: {dataset_path}")

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)                                    
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)

        train_dataset = Ecoset('train', dataset_path, train_transform, in_memory=in_memory) if 'train' in splits else None
        val_dataset = Ecoset('val', dataset_path, val_test_transform, in_memory=in_memory) if 'val' in splits else None
        test_dataset = Ecoset('test', dataset_path, val_test_transform, in_memory=in_memory) if 'test' in splits else None
 
        hyp['dataset']['num_classes'] = 112

    elif hyp['dataset']['name'] == 'ecoset_square256':
        

        dataset_path = f"{hyp['dataset']['dataset_path']}ecoset_square{hyp['dataset']['image_size']}_proper_chunks.h5"
        print(f"Loading dataset from: {dataset_path}")

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)                                    
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)

        train_dataset = Ecoset('train', dataset_path, train_transform, in_memory=in_memory) if 'train' in splits else None
        val_dataset = Ecoset('val', dataset_path, val_test_transform, in_memory=in_memory) if 'val' in splits else None
        test_dataset = Ecoset('test', dataset_path, val_test_transform, in_memory=in_memory) if 'test' in splits else None
        
        hyp['dataset']['num_classes'] = 565

    elif hyp['dataset']['name'] == 'ecoset_square256_patches':
        
        dataset_path = f"{hyp['dataset']['dataset_path']}optimized_datasets/megacoset.h5"
        print(f"Loading dataset from: {dataset_path}")

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)                                    
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)
            
        train_dataset = Ecoset('train', dataset_path, train_transform, in_memory=in_memory) if 'train' in splits else None
        val_dataset = Ecoset('val', dataset_path, val_test_transform, in_memory=in_memory) if 'val' in splits else None
        test_dataset = Ecoset('test', dataset_path, val_test_transform, in_memory=in_memory) if 'test' in splits else None
        
        hyp['dataset']['num_classes'] = 565
        

    elif hyp['dataset']['name'] == 'imagenet':
        from imagenet.imagenet import load_imagenet
        imagenet_path= "/share/klab/datasets/imagenet/"

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)   
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)
            
        train_loader, train_sampler, test_loader = load_imagenet(imagenet_path=imagenet_path,
                                                    batch_size=hyp['optimizer']['batch_size'],
                                                    distributed = False,
                                                    workers = hyp['optimizer']['dataloader']['num_workers_train'],
                                                    train_transforms = train_transform,
                                                    test_transforms= val_test_transform,
                                                    # normalization = False, # Not setting norm here
        )
        
        hyp['dataset']['num_classes'] = 1000
        hyp['dataset']['class_names'] = ["b'tench'", "b'goldfish'", "b'great_white_shark'", "b'tiger_shark'", "b'hammerhead'", "b'electric_ray'", "b'stingray'", "b'cock'", "b'hen'", "b'ostrich'", "b'brambling'", "b'goldfinch'", "b'house_finch'", "b'junco'", "b'indigo_bunting'", "b'robin'", "b'bulbul'", "b'jay'", "b'magpie'", "b'chickadee'", "b'water_ouzel'", "b'kite'", "b'bald_eagle'", "b'vulture'", "b'great_grey_owl'", "b'European_fire_salamander'", "b'common_newt'", "b'eft'", "b'spotted_salamander'", "b'axolotl'", "b'bullfrog'", "b'tree_frog'", "b'tailed_frog'", "b'loggerhead'", "b'leatherback_turtle'", "b'mud_turtle'", "b'terrapin'", "b'box_turtle'", "b'banded_gecko'", "b'common_iguana'", "b'American_chameleon'", "b'whiptail'", "b'agama'", "b'frilled_lizard'", "b'alligator_lizard'", "b'Gila_monster'", "b'green_lizard'", "b'African_chameleon'", "b'Komodo_dragon'", "b'African_crocodile'", "b'American_alligator'", "b'triceratops'", "b'thunder_snake'", "b'ringneck_snake'", "b'hognose_snake'", "b'green_snake'", "b'king_snake'", "b'garter_snake'", "b'water_snake'", "b'vine_snake'", "b'night_snake'", "b'boa_constrictor'", "b'rock_python'", "b'Indian_cobra'", "b'green_mamba'", "b'sea_snake'", "b'horned_viper'", "b'diamondback'", "b'sidewinder'", "b'trilobite'", "b'harvestman'", "b'scorpion'", "b'black_and_gold_garden_spider'", "b'barn_spider'", "b'garden_spider'", "b'black_widow'", "b'tarantula'", "b'wolf_spider'", "b'tick'", "b'centipede'", "b'black_grouse'", "b'ptarmigan'", "b'ruffed_grouse'", "b'prairie_chicken'", "b'peacock'", "b'quail'", "b'partridge'", "b'African_grey'", "b'macaw'", "b'sulphur-crested_cockatoo'", "b'lorikeet'", "b'coucal'", "b'bee_eater'", "b'hornbill'", "b'hummingbird'", "b'jacamar'", "b'toucan'", "b'drake'", "b'red-breasted_merganser'", "b'goose'", "b'black_swan'", "b'tusker'", "b'echidna'", "b'platypus'", "b'wallaby'", "b'koala'", "b'wombat'", "b'jellyfish'", "b'sea_anemone'", "b'brain_coral'", "b'flatworm'", "b'nematode'", "b'conch'", "b'snail'", "b'slug'", "b'sea_slug'", "b'chiton'", "b'chambered_nautilus'", "b'Dungeness_crab'", "b'rock_crab'", "b'fiddler_crab'", "b'king_crab'", "b'American_lobster'", "b'spiny_lobster'", "b'crayfish'", "b'hermit_crab'", "b'isopod'", "b'white_stork'", "b'black_stork'", "b'spoonbill'", "b'flamingo'", "b'little_blue_heron'", "b'American_egret'", "b'bittern'", "b'crane'", "b'limpkin'", "b'European_gallinule'", "b'American_coot'", "b'bustard'", "b'ruddy_turnstone'", "b'red-backed_sandpiper'", "b'redshank'", "b'dowitcher'", "b'oystercatcher'", "b'pelican'", "b'king_penguin'", "b'albatross'", "b'grey_whale'", "b'killer_whale'", "b'dugong'", "b'sea_lion'", "b'Chihuahua'", "b'Japanese_spaniel'", "b'Maltese_dog'", "b'Pekinese'", "b'Shih-Tzu'", "b'Blenheim_spaniel'", "b'papillon'", "b'toy_terrier'", "b'Rhodesian_ridgeback'", "b'Afghan_hound'", "b'basset'", "b'beagle'", "b'bloodhound'", "b'bluetick'", "b'black-and-tan_coonhound'", "b'Walker_hound'", "b'English_foxhound'", "b'redbone'", "b'borzoi'", "b'Irish_wolfhound'", "b'Italian_greyhound'", "b'whippet'", "b'Ibizan_hound'", "b'Norwegian_elkhound'", "b'otterhound'", "b'Saluki'", "b'Scottish_deerhound'", "b'Weimaraner'", "b'Staffordshire_bullterrier'", "b'American_Staffordshire_terrier'", "b'Bedlington_terrier'", "b'Border_terrier'", "b'Kerry_blue_terrier'", "b'Irish_terrier'", "b'Norfolk_terrier'", "b'Norwich_terrier'", "b'Yorkshire_terrier'", "b'wire-haired_fox_terrier'", "b'Lakeland_terrier'", "b'Sealyham_terrier'", "b'Airedale'", "b'cairn'", "b'Australian_terrier'", "b'Dandie_Dinmont'", "b'Boston_bull'", "b'miniature_schnauzer'", "b'giant_schnauzer'", "b'standard_schnauzer'", "b'Scotch_terrier'", "b'Tibetan_terrier'", "b'silky_terrier'", "b'soft-coated_wheaten_terrier'", "b'West_Highland_white_terrier'", "b'Lhasa'", "b'flat-coated_retriever'", "b'curly-coated_retriever'", "b'golden_retriever'", "b'Labrador_retriever'", "b'Chesapeake_Bay_retriever'", "b'German_short-haired_pointer'", "b'vizsla'", "b'English_setter'", "b'Irish_setter'", "b'Gordon_setter'", "b'Brittany_spaniel'", "b'clumber'", "b'English_springer'", "b'Welsh_springer_spaniel'", "b'cocker_spaniel'", "b'Sussex_spaniel'", "b'Irish_water_spaniel'", "b'kuvasz'", "b'schipperke'", "b'groenendael'", "b'malinois'", "b'briard'", "b'kelpie'", "b'komondor'", "b'Old_English_sheepdog'", "b'Shetland_sheepdog'", "b'collie'", "b'Border_collie'", "b'Bouvier_des_Flandres'", "b'Rottweiler'", "b'German_shepherd'", "b'Doberman'", "b'miniature_pinscher'", "b'Greater_Swiss_Mountain_dog'", "b'Bernese_mountain_dog'", "b'Appenzeller'", "b'EntleBucher'", "b'boxer'", "b'bull_mastiff'", "b'Tibetan_mastiff'", "b'French_bulldog'", "b'Great_Dane'", "b'Saint_Bernard'", "b'Eskimo_dog'", "b'malamute'", "b'Siberian_husky'", "b'dalmatian'", "b'affenpinscher'", "b'basenji'", "b'pug'", "b'Leonberg'", "b'Newfoundland'", "b'Great_Pyrenees'", "b'Samoyed'", "b'Pomeranian'", "b'chow'", "b'keeshond'", "b'Brabancon_griffon'", "b'Pembroke'", "b'Cardigan'", "b'toy_poodle'", "b'miniature_poodle'", "b'standard_poodle'", "b'Mexican_hairless'", "b'timber_wolf'", "b'white_wolf'", "b'red_wolf'", "b'coyote'", "b'dingo'", "b'dhole'", "b'African_hunting_dog'", "b'hyena'", "b'red_fox'", "b'kit_fox'", "b'Arctic_fox'", "b'grey_fox'", "b'tabby'", "b'tiger_cat'", "b'Persian_cat'", "b'Siamese_cat'", "b'Egyptian_cat'", "b'cougar'", "b'lynx'", "b'leopard'", "b'snow_leopard'", "b'jaguar'", "b'lion'", "b'tiger'", "b'cheetah'", "b'brown_bear'", "b'American_black_bear'", "b'ice_bear'", "b'sloth_bear'", "b'mongoose'", "b'meerkat'", "b'tiger_beetle'", "b'ladybug'", "b'ground_beetle'", "b'long-horned_beetle'", "b'leaf_beetle'", "b'dung_beetle'", "b'rhinoceros_beetle'", "b'weevil'", "b'fly'", "b'bee'", "b'ant'", "b'grasshopper'", "b'cricket'", "b'walking_stick'", "b'cockroach'", "b'mantis'", "b'cicada'", "b'leafhopper'", "b'lacewing'", "b'dragonfly'", "b'damselfly'", "b'admiral'", "b'ringlet'", "b'monarch'", "b'cabbage_butterfly'", "b'sulphur_butterfly'", "b'lycaenid'", "b'starfish'", "b'sea_urchin'", "b'sea_cucumber'", "b'wood_rabbit'", "b'hare'", "b'Angora'", "b'hamster'", "b'porcupine'", "b'fox_squirrel'", "b'marmot'", "b'beaver'", "b'guinea_pig'", "b'sorrel'", "b'zebra'", "b'hog'", "b'wild_boar'", "b'warthog'", "b'hippopotamus'", "b'ox'", "b'water_buffalo'", "b'bison'", "b'ram'", "b'bighorn'", "b'ibex'", "b'hartebeest'", "b'impala'", "b'gazelle'", "b'Arabian_camel'", "b'llama'", "b'weasel'", "b'mink'", "b'polecat'", "b'black-footed_ferret'", "b'otter'", "b'skunk'", "b'badger'", "b'armadillo'", "b'three-toed_sloth'", "b'orangutan'", "b'gorilla'", "b'chimpanzee'", "b'gibbon'", "b'siamang'", "b'guenon'", "b'patas'", "b'baboon'", "b'macaque'", "b'langur'", "b'colobus'", "b'proboscis_monkey'", "b'marmoset'", "b'capuchin'", "b'howler_monkey'", "b'titi'", "b'spider_monkey'", "b'squirrel_monkey'", "b'Madagascar_cat'", "b'indri'", "b'Indian_elephant'", "b'African_elephant'", "b'lesser_panda'", "b'giant_panda'", "b'barracouta'", "b'eel'", "b'coho'", "b'rock_beauty'", "b'anemone_fish'", "b'sturgeon'", "b'gar'", "b'lionfish'", "b'puffer'", "b'abacus'", "b'abaya'", "b'academic_gown'", "b'accordion'", "b'acoustic_guitar'", "b'aircraft_carrier'", "b'airliner'", "b'airship'", "b'altar'", "b'ambulance'", "b'amphibian'", "b'analog_clock'", "b'apiary'", "b'apron'", "b'ashcan'", "b'assault_rifle'", "b'backpack'", "b'bakery'", "b'balance_beam'", "b'balloon'", "b'ballpoint'", "b'Band_Aid'", "b'banjo'", "b'bannister'", "b'barbell'", "b'barber_chair'", "b'barbershop'", "b'barn'", "b'barometer'", "b'barrel'", "b'barrow'", "b'baseball'", "b'basketball'", "b'bassinet'", "b'bassoon'", "b'bathing_cap'", "b'bath_towel'", "b'bathtub'", "b'beach_wagon'", "b'beacon'", "b'beaker'", "b'bearskin'", "b'beer_bottle'", "b'beer_glass'", "b'bell_cote'", "b'bib'", "b'bicycle-built-for-two'", "b'bikini'", "b'binder'", "b'binoculars'", "b'birdhouse'", "b'boathouse'", "b'bobsled'", "b'bolo_tie'", "b'bonnet'", "b'bookcase'", "b'bookshop'", "b'bottlecap'", "b'bow'", "b'bow_tie'", "b'brass'", "b'brassiere'", "b'breakwater'", "b'breastplate'", "b'broom'", "b'bucket'", "b'buckle'", "b'bulletproof_vest'", "b'bullet_train'", "b'butcher_shop'", "b'cab'", "b'caldron'", "b'candle'", "b'cannon'", "b'canoe'", "b'can_opener'", "b'cardigan'", "b'car_mirror'", "b'carousel'", "b'carpenter's_kit'", "b'carton'", "b'car_wheel'", "b'cash_machine'", "b'cassette'", "b'cassette_player'", "b'castle'", "b'catamaran'", "b'CD_player'", "b'cello'", "b'cellular_telephone'", "b'chain'", "b'chainlink_fence'", "b'chain_mail'", "b'chain_saw'", "b'chest'", "b'chiffonier'", "b'chime'", "b'china_cabinet'", "b'Christmas_stocking'", "b'church'", "b'cinema'", "b'cleaver'", "b'cliff_dwelling'", "b'cloak'", "b'clog'", "b'cocktail_shaker'", "b'coffee_mug'", "b'coffeepot'", "b'coil'", "b'combination_lock'", "b'computer_keyboard'", "b'confectionery'", "b'container_ship'", "b'convertible'", "b'corkscrew'", "b'cornet'", "b'cowboy_boot'", "b'cowboy_hat'", "b'cradle'", "b'crane'", "b'crash_helmet'", "b'crate'", "b'crib'", "b'Crock_Pot'", "b'croquet_ball'", "b'crutch'", "b'cuirass'", "b'dam'", "b'desk'", "b'desktop_computer'", "b'dial_telephone'", "b'diaper'", "b'digital_clock'", "b'digital_watch'", "b'dining_table'", "b'dishrag'", "b'dishwasher'", "b'disk_brake'", "b'dock'", "b'dogsled'", "b'dome'", "b'doormat'", "b'drilling_platform'", "b'drum'", "b'drumstick'", "b'dumbbell'", "b'Dutch_oven'", "b'electric_fan'", "b'electric_guitar'", "b'electric_locomotive'", "b'entertainment_center'", "b'envelope'", "b'espresso_maker'", "b'face_powder'", "b'feather_boa'", "b'file'", "b'fireboat'", "b'fire_engine'", "b'fire_screen'", "b'flagpole'", "b'flute'", "b'folding_chair'", "b'football_helmet'", "b'forklift'", "b'fountain'", "b'fountain_pen'", "b'four-poster'", "b'freight_car'", "b'French_horn'", "b'frying_pan'", "b'fur_coat'", "b'garbage_truck'", "b'gasmask'", "b'gas_pump'", "b'goblet'", "b'go-kart'", "b'golf_ball'", "b'golfcart'", "b'gondola'", "b'gong'", "b'gown'", "b'grand_piano'", "b'greenhouse'", "b'grille'", "b'grocery_store'", "b'guillotine'", "b'hair_slide'", "b'hair_spray'", "b'half_track'", "b'hammer'", "b'hamper'", "b'hand_blower'", "b'hand-held_computer'", "b'handkerchief'", "b'hard_disc'", "b'harmonica'", "b'harp'", "b'harvester'", "b'hatchet'", "b'holster'", "b'home_theater'", "b'honeycomb'", "b'hook'", "b'hoopskirt'", "b'horizontal_bar'", "b'horse_cart'", "b'hourglass'", "b'iPod'", "b'iron'", "b'jack-o'-lantern'", "b'jean'", "b'jeep'", "b'jersey'", "b'jigsaw_puzzle'", "b'jinrikisha'", "b'joystick'", "b'kimono'", "b'knee_pad'", "b'knot'", "b'lab_coat'", "b'ladle'", "b'lampshade'", "b'laptop'", "b'lawn_mower'", "b'lens_cap'", "b'letter_opener'", "b'library'", "b'lifeboat'", "b'lighter'", "b'limousine'", "b'liner'", "b'lipstick'", "b'Loafer'", "b'lotion'", "b'loudspeaker'", "b'loupe'", "b'lumbermill'", "b'magnetic_compass'", "b'mailbag'", "b'mailbox'", "b'maillot'", "b'maillot'", "b'manhole_cover'", "b'maraca'", "b'marimba'", "b'mask'", "b'matchstick'", "b'maypole'", "b'maze'", "b'measuring_cup'", "b'medicine_chest'", "b'megalith'", "b'microphone'", "b'microwave'", "b'military_uniform'", "b'milk_can'", "b'minibus'", "b'miniskirt'", "b'minivan'", "b'missile'", "b'mitten'", "b'mixing_bowl'", "b'mobile_home'", "b'Model_T'", "b'modem'", "b'monastery'", "b'monitor'", "b'moped'", "b'mortar'", "b'mortarboard'", "b'mosque'", "b'mosquito_net'", "b'motor_scooter'", "b'mountain_bike'", "b'mountain_tent'", "b'mouse'", "b'mousetrap'", "b'moving_van'", "b'muzzle'", "b'nail'", "b'neck_brace'", "b'necklace'", "b'nipple'", "b'notebook'", "b'obelisk'", "b'oboe'", "b'ocarina'", "b'odometer'", "b'oil_filter'", "b'organ'", "b'oscilloscope'", "b'overskirt'", "b'oxcart'", "b'oxygen_mask'", "b'packet'", "b'paddle'", "b'paddlewheel'", "b'padlock'", "b'paintbrush'", "b'pajama'", "b'palace'", "b'panpipe'", "b'paper_towel'", "b'parachute'", "b'parallel_bars'", "b'park_bench'", "b'parking_meter'", "b'passenger_car'", "b'patio'", "b'pay-phone'", "b'pedestal'", "b'pencil_box'", "b'pencil_sharpener'", "b'perfume'", "b'Petri_dish'", "b'photocopier'", "b'pick'", "b'pickelhaube'", "b'picket_fence'", "b'pickup'", "b'pier'", "b'piggy_bank'", "b'pill_bottle'", "b'pillow'", "b'ping-pong_ball'", "b'pinwheel'", "b'pirate'", "b'pitcher'", "b'plane'", "b'planetarium'", "b'plastic_bag'", "b'plate_rack'", "b'plow'", "b'plunger'", "b'Polaroid_camera'", "b'pole'", "b'police_van'", "b'poncho'", "b'pool_table'", "b'pop_bottle'", "b'pot'", "b'potter's_wheel'", "b'power_drill'", "b'prayer_rug'", "b'printer'", "b'prison'", "b'projectile'", "b'projector'", "b'puck'", "b'punching_bag'", "b'purse'", "b'quill'", "b'quilt'", "b'racer'", "b'racket'", "b'radiator'", "b'radio'", "b'radio_telescope'", "b'rain_barrel'", "b'recreational_vehicle'", "b'reel'", "b'reflex_camera'", "b'refrigerator'", "b'remote_control'", "b'restaurant'", "b'revolver'", "b'rifle'", "b'rocking_chair'", "b'rotisserie'", "b'rubber_eraser'", "b'rugby_ball'", "b'rule'", "b'running_shoe'", "b'safe'", "b'safety_pin'", "b'saltshaker'", "b'sandal'", "b'sarong'", "b'sax'", "b'scabbard'", "b'scale'", "b'school_bus'", "b'schooner'", "b'scoreboard'", "b'screen'", "b'screw'", "b'screwdriver'", "b'seat_belt'", "b'sewing_machine'", "b'shield'", "b'shoe_shop'", "b'shoji'", "b'shopping_basket'", "b'shopping_cart'", "b'shovel'", "b'shower_cap'", "b'shower_curtain'", "b'ski'", "b'ski_mask'", "b'sleeping_bag'", "b'slide_rule'", "b'sliding_door'", "b'slot'", "b'snorkel'", "b'snowmobile'", "b'snowplow'", "b'soap_dispenser'", "b'soccer_ball'", "b'sock'", "b'solar_dish'", "b'sombrero'", "b'soup_bowl'", "b'space_bar'", "b'space_heater'", "b'space_shuttle'", "b'spatula'", "b'speedboat'", "b'spider_web'", "b'spindle'", "b'sports_car'", "b'spotlight'", "b'stage'", "b'steam_locomotive'", "b'steel_arch_bridge'", "b'steel_drum'", "b'stethoscope'", "b'stole'", "b'stone_wall'", "b'stopwatch'", "b'stove'", "b'strainer'", "b'streetcar'", "b'stretcher'", "b'studio_couch'", "b'stupa'", "b'submarine'", "b'suit'", "b'sundial'", "b'sunglass'", "b'sunglasses'", "b'sunscreen'", "b'suspension_bridge'", "b'swab'", "b'sweatshirt'", "b'swimming_trunks'", "b'swing'", "b'switch'", "b'syringe'", "b'table_lamp'", "b'tank'", "b'tape_player'", "b'teapot'", "b'teddy'", "b'television'", "b'tennis_ball'", "b'thatch'", "b'theater_curtain'", "b'thimble'", "b'thresher'", "b'throne'", "b'tile_roof'", "b'toaster'", "b'tobacco_shop'", "b'toilet_seat'", "b'torch'", "b'totem_pole'", "b'tow_truck'", "b'toyshop'", "b'tractor'", "b'trailer_truck'", "b'tray'", "b'trench_coat'", "b'tricycle'", "b'trimaran'", "b'tripod'", "b'triumphal_arch'", "b'trolleybus'", "b'trombone'", "b'tub'", "b'turnstile'", "b'typewriter_keyboard'", "b'umbrella'", "b'unicycle'", "b'upright'", "b'vacuum'", "b'vase'", "b'vault'", "b'velvet'", "b'vending_machine'", "b'vestment'", "b'viaduct'", "b'violin'", "b'volleyball'", "b'waffle_iron'", "b'wall_clock'", "b'wallet'", "b'wardrobe'", "b'warplane'", "b'washbasin'", "b'washer'", "b'water_bottle'", "b'water_jug'", "b'water_tower'", "b'whiskey_jug'", "b'whistle'", "b'wig'", "b'window_screen'", "b'window_shade'", "b'Windsor_tie'", "b'wine_bottle'", "b'wing'", "b'wok'", "b'wooden_spoon'", "b'wool'", "b'worm_fence'", "b'wreck'", "b'yawl'", "b'yurt'", "b'web_site'", "b'comic_book'", "b'crossword_puzzle'", "b'street_sign'", "b'traffic_light'", "b'book_jacket'", "b'menu'", "b'plate'", "b'guacamole'", "b'consomme'", "b'hot_pot'", "b'trifle'", "b'ice_cream'", "b'ice_lolly'", "b'French_loaf'", "b'bagel'", "b'pretzel'", "b'cheeseburger'", "b'hotdog'", "b'mashed_potato'", "b'head_cabbage'", "b'broccoli'", "b'cauliflower'", "b'zucchini'", "b'spaghetti_squash'", "b'acorn_squash'", "b'butternut_squash'", "b'cucumber'", "b'artichoke'", "b'bell_pepper'", "b'cardoon'", "b'mushroom'", "b'Granny_Smith'", "b'strawberry'", "b'orange'", "b'lemon'", "b'fig'", "b'pineapple'", "b'banana'", "b'jackfruit'", "b'custard_apple'", "b'pomegranate'", "b'hay'", "b'carbonara'", "b'chocolate_sauce'", "b'dough'", "b'meat_loaf'", "b'pizza'", "b'potpie'", "b'burrito'", "b'red_wine'", "b'espresso'", "b'cup'", "b'eggnog'", "b'alp'", "b'bubble'", "b'cliff'", "b'coral_reef'", "b'geyser'", "b'lakeside'", "b'promontory'", "b'sandbar'", "b'seashore'", "b'valley'", "b'volcano'", "b'ballplayer'", "b'groom'", "b'scuba_diver'", "b'rapeseed'", "b'daisy'", "b'yellow_lady's_slipper'", "b'corn'", "b'acorn'", "b'hip'", "b'buckeye'", "b'coral_fungus'", "b'agaric'", "b'gyromitra'", "b'stinkhorn'", "b'earthstar'", "b'hen-of-the-woods'", "b'bolete'", "b'ear'", "b'toilet_tissue'"]
        
        return train_loader, test_loader, test_loader, hyp
    

    elif hyp['dataset']['name'] == 'facescrub':
        dataset_file = '/share/klab/datasets/texture2shape_projects/generate_facescrub_dataset/facescrub_256px.h5'
        print(f"Loading dataset from: {dataset_file}")

        with h5py.File(dataset_file, "r") as f:
            hyp['dataset']['num_classes'] = np.array(f['categories']).shape[0]
        
        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)  
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)
            
        train_dataset = MiniEcoset('train', dataset_file, train_transform)
        val_dataset = MiniEcoset('val', dataset_file, val_test_transform)
        test_dataset = MiniEcoset('test', dataset_file, val_test_transform)

    elif hyp['dataset']['name'] == 'STL10':
        # Good for debugging
        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)  
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)
        
            
        train_dataset = torchvision.datasets.STL10(
            './datasets/',
            split="train", #* unlabeled now for SSL, 'train' for supervised
            download=True,
            transform = train_transform,     
        )
        val_dataset = torchvision.datasets.STL10(
            './datasets/',
            split="train", # train with label but val without labels
            download=True,
            transform = val_test_transform,     
        )
        test_dataset = torchvision.datasets.STL10(
            './datasets/',
            split="test",
            download=True,
            transform = val_test_transform,     
        )

        hyp['dataset']['num_classes'] = 10

    else:
        raise ValueError(f"Unknown dataset: {hyp['dataset']['name']}")

    # Create Dataloaders for the splits
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyp['optimizer']['batch_size'], shuffle=True,
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_train'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_train'],
                                                # pin_memory=False,                # <--- recommended True for CUDA
                                                # persistent_workers=False         # <--- recommended
                                                ) if 'train' in splits else None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'],
                                                # pin_memory=False,                # <--- recommended True for CUDA
                                                # persistent_workers=False         # <--- recommended
                                                ) if 'val' in splits else None
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'],
                                                # pin_memory=False,                # <--- recommended True for CUDA
                                                # persistent_workers=False         # <--- recommended
                                                ) if 'test' in splits else None


    
    if compute_stats:
        mean, std = compute_mean_std(train_loader)
        print(f"train loader mean & std: {mean}, {std}")
        mean, std = compute_mean_std(val_loader)
        print(f"val loader mean & std: {mean}, {std}")
        mean, std = compute_mean_std(test_loader)
        print(f"test loader mean & std: {mean}, {std}")  
        
    return train_loader, val_loader, test_loader, hyp


def get_transform(aug_str, hyp=None, normalize_type='0-1'):
    """
    Build a Kornia augmentation pipeline as an nn.Module.
    All transforms will run on GPU if the input tensor is on GPU.
    """
    aug_list = []

    # ADD to float, need to be done before Konia transforms
    aug_list.append(torchvision.transforms.ConvertImageDtype(torch.float))

    # # Example: resize to 224 x 224
    # if 'resize' in aug_str:
    #     # Kornia uses (height, width)
    #     aug_list.append(K.Resize((224, 224), align_corners=True))

    # Random horizontal flip
    if 'randomflip' in aug_str:
        aug_list.append(K.RandomHorizontalFlip(p=0.5))
    # Random rotation
    if 'randomrotation' in aug_str:
        # degrees=45 => rotate in [-45, +45]
        aug_list.append(K.RandomRotation(degrees=45.0, p=0.5))
    # Random grayscale
    if 'grayscale' in aug_str:
        aug_list.append(K.RandomGrayscale(p=0.5))
    # Random brightness, equalize, perspective
    if 'globalbrightness' in aug_str:
        # brightness=(0.8, 1.2)
        aug_list.append(K.RandomBrightness(brightness=(0.8, 1.2), p=0.5))
    if 'equalize' in aug_str:
        aug_list.append(K.RandomEqualize(p=0.5))
    if 'perspective' in aug_str:
        aug_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.5))

    
    # Build the final pipeline
    augmentations = nn.Sequential(*aug_list)


    # For example:
    class KorniaTransform(nn.Module):
        def __init__(self, aug_pipe, normalize_type, hyp):
            super().__init__()
            self.aug_pipe = aug_pipe
            self.normalize_type = normalize_type
            
            if hyp is not None and 'train_img_mean_channels' in hyp:
                self.mean = torch.tensor(hyp['train_img_mean_channels']) / 255.
                self.std = torch.tensor(hyp['train_img_std_channels']) / 255.
            else:
                self.mean = torch.tensor([0.485, 0.456, 0.406])
                self.std = torch.tensor([0.229, 0.224, 0.225])
                
        def forward(self, x):

            x = torchvision.transforms.functional.to_tensor(x)
            # x should be a Tensor (B,C,H,W) on GPU
            x = self.aug_pipe(x)
            # optional normalization
            if 'normalize' in aug_str:
                if self.normalize_type == '0-1':
                    pass  # already in 0..1 if x is in that range
                elif self.normalize_type == 'mean-std':
                    # Kornia has a normalize as well
                    x = K.Normalize(mean=self.mean, std=self.std)(x)
                elif self.normalize_type == '-1-1':
                    # shift from [0,1] to [-1,1]
                    # x = (x - 0.5) / 0.5
                    half = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, -1, 1, 1)
                    x = (x - half) / half
                else:
                    raise ValueError(f"Unknown normalize type: {self.normalize_type}")
            return x

    return KorniaTransform(augmentations, normalize_type, hyp)

def compute_mean_std(loader):
    # Variables for the sum and square sum of all pixels and the number of batches
    mean = 0.0
    mean_square = 0.0
    samples = 0

    for data in loader:
        # Assuming the data loader returns a tuple of (images, labels)
        images, _ = data
        # If images is a list (self-supervised setting), combine the two views
        if isinstance(images, list):
            images = torch.cat(images, dim=0)  # Concatenate along the batch dimension
            
        # Flatten the channels
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        mean_square += (images ** 2).mean(2).sum(0)
        samples += images.size(0)
    
    # Final calculation of mean and std
    mean /= samples
    mean_square /= samples
    std = (mean_square - mean ** 2) ** 0.5

    return mean, std



class Ecoset(torch.utils.data.Dataset):
    #Import Ecoset as a Dataset splitwise

    def __init__(self, split, dataset_path, transform=None, in_memory=False, ):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.
            in_memory: Should we pre-load the dataset?
        """
        self.root_dir = dataset_path
        self.transform = transform
        self.split = split
        self.in_memory = in_memory

        if self.in_memory:
            with h5py.File(dataset_path, "r") as f:
                self.images = torch.from_numpy(f[split]['data'][()]).permute((0, 3, 1, 2)) # to match the CHW expectation of pytorch
                self.labels = torch.from_numpy(f[split]['labels'][()].astype(np.int64)).long()
        else:
            self.split_data = h5py.File(dataset_path, "r")[split]
            self.images = self.split_data['data']
            self.labels = self.split_data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): # accepts ids and returns the images and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.in_memory:
            imgs = self.images[idx]
            labels = self.labels[idx]
        else:
            with h5py.File(self.root_dir, "r") as f:
                imgs = torch.from_numpy(np.asarray(self.images[idx])).permute((2,0,1))    
                labels = torch.from_numpy(np.asarray(self.labels[idx].astype(np.int64))).long()

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]
