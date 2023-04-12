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
    '¥kland',
    'AJ',
    'Altibox',
    'Bama',
    'Borregaard',
    'Bouvet',
    'Cegal',
    'Consto',
    'Coop',
    'DNB',
    'Equinor',
    'Fjordkraft',
    'Frydenb¢',
    'Gjensidige',
    'Kiwi',
    'Lyse',
    'NorskTipping',
    'OBOS',
    'Santander',
    'Pretec',
    'Scandic',
    'Sparebank1 Nord Norge',
    'Sparebank1 SMN',
    'Sparebanken Møre',
    'Sparebanken Vest',
    'Sparebank1 SR Bank',
    'Sundolitt',
    'Telenor',
    'Tine',
    'Vaerste',
    'Vanpee',
    'Sparebank1 group',
]

# create own Dataset
my_dataset = simpleDataset(root=root,
                           filenames=filenames,
                           labels=labels
                           )

