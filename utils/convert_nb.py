'''
To convert a jupyter notebook into a markdown post with the correct image referencing

Dependencies:

* os
* re
* argparse
* nbconvert
* PIL
'''

import os
import re
import argparse
import nbconvert
from nbconvert import MarkdownExporter
from multiprocessing import Pool
from PIL import Image
import io
import datetime

from nbconvert import nbconvertapp

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def save_image(self,k):
    # k is the key
    image = Image.open(io.BytesIO(self.resources['outputs'][k]))
    image.save(self.blogpath+k)

class convert_notebook(object):
    '''
    Converts notebook for the blog.

    Inputs .ipynb, outputs .md

    Markdown corrected with images to

    Usage:

    foo = convert_notebook(notebook_path,blog="blog_root_directory")
    foo.convert()
    '''

    def __init__(self,notebook_path,title,description,execute=False,type="posts",blogpath="~/Sites/iitmcvg.github.io"):

        '''
        initialise object
        run nbconvert extraction and saving files
        '''

        if execute:
            # Execute notebook
            with open(notebook_path) as f:
                nb = nbformat.read(f, as_version=4)

            try:
                out = ep.preprocess(nb, {'metadata': {'path': run_path}})
            except CellExecutionError:
                # If error in execution
                out = None
                msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
                msg += 'See notebook "%s" for the traceback.' % notebook_filename_out
                print(msg)
                raise
            finally:
                notebook_path=notebook_path[:-6]+'executed_notebook.ipynb'

                with open(notebook_filename_out, mode='wt') as f:
                    nbformat.write(nb, f)

        # The extension
        self.extension="ipynb"

        # The name with the extension
        self.notebook=re.search(r'[a-zA-Z0-9_(\ )]*.ipynb',notebook_path).group()

        # Just the name
        self.notebook_name=re.search(r'[a-zA-Z0-9_(\ )]*.ipynb',notebook_path).group()[:-len(self.extension)-1]
        self.notebook_path=notebook_path

        self.type=type

        # Python doesnt recognise ~ at times
        self.blogpath=os.path.expanduser(blogpath)
        self.title=title
        self.description=description

        self.get_output_resources()
        self.save_images()
        self.process_output()

    def get_output_resources(self,output_path=None):
        '''
        Markdown Exporter
        '''
        if output_path is None:
            output_path=os.path.join(os.path.sep,"assets","images",self.type,self.notebook_name)
            try:
                # We are creating nested directories, hence
                os.makedirs(self.blogpath+output_path)
            except:
                pass
        self.output_path=self.blogpath+"/"+output_path
        md=MarkdownExporter()

        # Extract config dictionary
        nbapp=nbconvertapp.NbConvertApp()
        self.config=nbapp.init_single_notebook_resources(self.notebook_path)
        self.config['output_files_dir']=output_path

        '''
        Of type:

        {'config_dir': '/Users/Ankivarun/.jupyter', \
         'unique_key': self.notebook_name,\
         'output_files_dir': output_path}
        '''

        self.output,self.resources=md.from_filename(self.notebook_path,self.config)

    def save_images(self):
        '''
        Multiprocessed implementation for saving images.
        '''
        print(self.resources['outputs'].keys())
        with Pool() as pool:
            pool.starmap(save_image, zip([self for j in range(len(self.resources['outputs']))],\
            self.resources['outputs'].keys()))

    def process_output(self):
        '''
        Processing into Jekyll Templates
        '''

        # Latex pre-processing
        regex = re.compile(r'\$|\$\$')
        self.output=regex.sub("$$", self.output)

        regex= re.compile(r'\\begin{equation}')
        self.output=regex.sub(r"$$ \\begin{equation}", self.output)

        regex= re.compile(r'\\end{equation}')
        self.output=regex.sub(r"\\end{equation} $$", self.output)

        preamble=r"""---
layout: single
title: " """+self.title+ """ "
category: machine_learning
description: \" """+self.description + """ \"
mathjax: true
---"""

        self.output=preamble+self.output

        # Write markdown file
        if self.type=="posts":
            today=datetime.datetime.today()
            year=str(today.year)
            month=str(today.month)
            day=str(today.day)
            # If it is a post, save the date
            title="-".join([year,month,day])+"-"+self.notebook_name+".md"
            with open(os.path.join(self.blogpath,"_"+self.type,title),'w+') as f:
                f.write(self.output)
        else:
            with open(os.path.join(self.blogpath,"_"+self.type,self.notebook_name+".md"),'w+') as f:
                f.write(self.output)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=" Convert .ipynb notebook into a blog post")
    parser.add_argument("-nb","--notebook",help="notebook path",required=True)
    parser.add_argument("-t","--title",help="Title of article [post|page|any class type]",required=True)
    parser.add_argument("-d","--description",help="Description of article",required=True)
    parser.add_argument("-e","--execute",help="Execute the jupyter notebook before converting",action='store_const', const=1,default=False)
    parser.add_argument("-ty","--type",help="notebook type",default="posts")
    parser.add_argument("-bp","--blogpath",help="Blogsite path",default="~/Sites/iitmcvg.github.io")
    args=parser.parse_args()
    notebook_path=args.notebook
    title=args.title
    description=args.description
    type=args.type
    blogpath=args.blogpath
    execute=args.execute
    conv=convert_notebook(notebook_path=notebook_path,title=title,description=description\
    ,execute=execute,type=type,blogpath=blogpath)
