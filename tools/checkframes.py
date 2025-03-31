from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path
import shutil


def draw(images_path, labels_path):
    pa = Path.cwd() / 'video-test' / 'dt-tomato'
    pal = Path(pa / labels_path)
    # dir_list = glob.glob(os.path.join(f'{pa}/{labels_path}', f'*.txt'))
    dir_list = sorted(pal.glob('*'))
    for each in dir_list:
        check(pa, images_path, pal, os.path.splitext(os.path.split(each)[1])[0])


def check(path, images_path, labels_path, file_name):
    im = Image.open(f'{path}/{images_path}/{file_name}.jpg')
    width, height = im.size
    with open(f"{labels_path}/{file_name}.txt") as file:
        fl = file.readlines()[0:]
        for line in fl:
            a = line.split()
            x = float(a[1]) * width
            y = float(a[2]) * height
            wf = float(a[3]) * width
            hf = float(a[4]) * height
            myFont = ImageFont.truetype('arial.ttf', 15)
            ImageDraw.Draw(im).text((x, y), a[0], font=myFont, fill=(255, 0, 0))
            draw = ImageDraw.Draw(im)
            draw.rectangle(((x - wf / 2), (y - hf / 2), (x + wf / 2), (y + hf / 2)), outline=(255, 255, 255), width=2)

        im.save(f'{path}/t{file_name}.jpg', quality=95)


if __name__ == '__main__':
    draw('images', 'lables')


    # # Укажите путь к папке, которую нужно архивировать
    # folder_path = Path.cwd() / 'data' / 'Helmetxy'

    # # Укажите путь и имя архива
    # archive_path = Path.cwd() / 'data' / 'Helmetxy'

    # # Создайте архив
    # shutil.make_archive(archive_path, 'zip', folder_path)
