import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from options.test_options import TestOptions
from data import create_dataset
from Mroph import create_model
from util.visualizer import save_images
from util import html
import time

if __name__ == '__main__':
    opt = TestOptions().parse()

    opt.num_threads = 1
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True
    opt.display_id = -1
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.print_networks(opt.verbose)
    model.setup(opt)
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        start_time = time.time()
        model.test()
        stop_time = time.time()
        print("Test_time:", stop_time - start_time, "seconds")
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, opt.img_spacing,aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
