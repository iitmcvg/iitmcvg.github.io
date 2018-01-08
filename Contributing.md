# Contribution Guidelines

As a student run club, we are always enthusiastic about open-source and community contributions. Our objective has always been towards making tools of computer vision and deep learning more accessible and deployable in various situations- so we would want you to keep this in mind.

---

## Copyright and Proprietary Content

At present, our philosophy is to keep the site repository public and accessible for pull requests - in view of growing a community for Computer Vision and Intelligence.

Should your post contain any matter that is either proprietary or held under private intellectual rights, we would request you to link such matter with external cloud hosting (example: onedrive for model weights and checkpoints). This will let you have control over the usage of such material.

In case certain portions of your post happen to be inspired directly from any other available material, do suitably credit the original source. At CVI, we are committed to adding value to the Deep Learning community, while ensuring that diligence paid in crediting respective authors.

---

## Uploading binary and static files

_Note: all files excluding .md, .yml,.toml files and image files come under the purview of static files for this article_

Since we are hosting our site with Github pages, we would like to follow a BNBR (Be Nice, Be Respectful) policy of not exceeding a said limit size for the repository.

Hence, in order to display larger content (any multimedia or other files) we recommend sending in a public link after using a cloud file service. In particular for weight files, checkpoints and tensorflow graphfiles we are absolutely keen that any such content is not directly added to the repository.

---

## Sending in a Post

Go ahead, submit a pull request with the following:

* Your post under _drafts/
* A suitable title under the _frontmatter variable: title_
* Suitable front matter variable : toc, if you would like a table of contents.
* Once we publish your post, we would move it to the _posts/ directory with relevant requirements.

### Previewing your post locally

You can preview your post locally if you wish to, even prior to submitting a pull request. This can be handy to check for any formatting or styling issues. In which case, here are the steps:

* Install [Jekyll](https://jekyllrb.com/)
* Run `bundle exec jekyll serve --drafts`
* You should now be able to view the local site under localhost:4000 (or similar).
* Incase you want to edit multiple times, it might be useful to enable the `--incremental` flag, especially as our site contiues to grow.

---

## Converting a Jupyter Notebook for a Post

Jupyter notebooks offer a lot of flexibility when it comes to elucidating on code, and as such we are happy to host your content which you may have in form of .ipynb notebooks.

You may follow these steps to convert your jupyter notebook content:

* Use jupyter [nbconvert](https://github.com/jupyter/nbconvert.git), a nifty tool for exporting jupyter notebooks.
* Run `jupyter nbconvert --to md <input notebook>`
* Place the generated media (images, plot files), under the /assets/images/posts/<name of post> directory.
