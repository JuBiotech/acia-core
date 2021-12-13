from acia.base import ImageRoISource
from acia.segm.local import LocalSequenceSource, ImageJRoISource
from acia.segm.output import CocoDataset

if __name__ == '__main__':
    train_irs_list = [ImageRoISource(LocalSequenceSource('sim.tif'), ImageJRoISource('sim.tif'))]
    val_irs_list = [ImageRoISource(LocalSequenceSource('val.tif'), ImageJRoISource('val.tif'))]

    print("Downloading Training dataset...")
    cd = CocoDataset()
    cd.add(train_irs_list)
    cd.write('coco')

    print("Downloading Validation dataset...")
    cd = CocoDataset()
    cd.add(val_irs_list)
    cd.write('coco', 'val')
