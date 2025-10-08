import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
import random

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path,img_spacing,aspect_ratio=1.0, width=128):

    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = util.tensor2im(im_data, label)
        image_numpy_tranpose = np.transpose(im, (3, 0, 1, 2))
        image_numpy_tranpose_gray = image_numpy_tranpose[0][:][:][:]

        image_name = '%s_%s.nii' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        h, w, c, _ = im.shape

        util.save_image(image_numpy_tranpose_gray, save_path, img_spacing)

        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    webpage.add_images(ims, txts, links, width=width)


class Visualizer():

    def __init__(self, opt):
        self.opt = opt
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False
        self.spacing = opt.img_spacing
        self.img_spacing= self.spacing
        if self.display_id > 0:
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def create_visdom_connections(self):
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):

        if self.display_id > 0:
            ncols = self.ncols
            if ncols > 0:
                ncols = min(ncols, len(visuals))

                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)
                title = self.name
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                image_numpy_to_x=util.tensor2im(visuals["real_E_1"],"real_E_1")
                image_numpy_to_x_1=image_numpy_to_x.transpose([3,0,1,2])
                x = int(image_numpy_to_x_1.shape[1]/2)
                for label, image in visuals.items():
                    image_numpy = util.tensor2im(image, label)
                    label_html_row += '<td>%s</td>' % label

                    image_numpy_1 = image_numpy.transpose([3, 0, 1, 2])
                    # if label == "real_C":
                    #     x = int(np.median(np.array(np.where(image_numpy_1 != 0))[1]))
                    image_toshow = image_numpy_1[0, x, :, :]
                    # print("label:{},image_numpy.shape{},image_numpy_1.shape{},image_toshow.shape{}".format(label,image_numpy.shape,image_numpy_1.shape,image_toshow.shape))
                    # print("1",x,np.max(image_numpy_1),np.min(image_numpy_1))

                    images.append(image_toshow)

                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                images_TobeShow = np.concatenate(images,axis=1)
                white_image = np.ones_like(image_toshow) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images_TobeShow, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    # self.vis.images(images, nrow=8, win=self.display_id + 1,
                    #                 padding=2, opts=dict(title=title + ' images'))
                    # print("len(images)",len(images))
                    # print(ncols)
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:
                idx = 1
                try:
                    image_numpy_to_x = util.tensor2im(visuals["real_E_1"],"real_E_1")
                    image_numpy_to_x_1 = image_numpy_to_x.transpose([3, 0, 1, 2])
                    x = int(np.median(np.array(np.where(image_numpy_to_x_1 != 0))[1]))
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)

                        image_numpy_1 = image_numpy.transpose([3, 0, 1, 2])
                        # if label == "real_C":
                        #     x = int(np.median(np.array(np.where(image_numpy_1 != 0))[1]))
                        image_toshow = image_numpy_1[0, x, :, :]
                        # print("2", x, np.max(image_numpy_1), np.min(image_numpy_1))

                        self.vis.image(image_toshow, opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()

        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            for label, image in visuals.items():
                # type(image) = <class 'torch.Tensor'>
                image_numpy = util.tensor2im(image, label)

                image_numpy_tranpose = np.transpose(image_numpy, (3, 0, 1, 2))
                image_numpy_tranpose_gray = image_numpy_tranpose[0][:][:][:]
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.nii' % (epoch, label))
                # print("spacing",self.img_spacing)
                util.save_image(image_numpy_tranpose_gray, img_path, self.img_spacing)


            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=1)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():
                    image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.png' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k] for k in self.plot_data['legend']])
        try:
            self.vis.line(
                X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
