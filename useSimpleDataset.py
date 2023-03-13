from simpleDataset import simpleDataset

# data directory


root = "dataset1/data"

number_of_images = 1000
filenames = []

for i in range(number_of_images):
    filename = str(i) + ".jpg"
    filenames.append(filename)


# assume we have 3 jpg images

# the class of image might be ['black cat', 'tabby cat', 'tabby cat']
labels = [
    'Sparebanken Vest',
    'Pretec',
    'Borregaard',
    'OBOS',
    'Lyse',
    'Sparebank1 SR Bank',
    'Cegal',
    'Bouvet',
    'Coop',
    'Sundolitt',
    'AJ',
    'DNB',
    'Vanpee',
    '¥kland',
    'Vaerste',
    'Altibox',
    'NorskTipping',
    'Bama',
    'Tine',
    'Telenor',
    'Sparebank1 SMN',
    'Scandic',
    'Fjordkraft',
    'Gjensidige',
    'Frydenb¢',
    'Consto',
    'Sparebank1 Nord Norge',
    'Kiwi',
    'Equinor',
    'Santander',
    'Sparebanken Møre',
    'Sparebank1 group'
]

# create own Dataset
my_dataset = simpleDataset(root=root,
                           filenames=filenames,
                           labels=labels
                           )