#!/bin/bash

echo "Begining by updating and upgrading your OS"

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  echo "Linux Platform Detected"
  sudo apt-get -y update
  sudo apt-get -y upgrade
  sudo apt-get -y install ruby ruby-dev build-essential

  echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
  echo 'export GEM_HOME=$HOME/gems' >> ~/.bashrc
  echo 'export PATH=$HOME/gems/bin:$PATH' >> ~/.bashrc
  source ~/.bashrc

  echo '# Install Ruby Gems to ~/gems' >> ~/.zshrc
  echo 'export GEM_HOME=$HOME/gems' >> ~/.zshrc
  echo 'export PATH=$HOME/gems/bin:$PATH' >> ~/.zshrc
  source ~/.zshrc

  gem install jekyll bundler

elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "Mac OS Platform Detected"
  # Check if Homebrew is installed
  which -s brew
  if [[ $? != 0 ]] ; then
    echo "Installing homebrew"
    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

  else
    echo "Homebrew Detected"
    echo "Updating Brew"
    brew update
  fi

  brew install ruby
  ruby -v


  xcode-select --install

  echo "Installing bundler"
  gem install bundler
  gem install bundler jekyll

  # Add Jekyll
  bundle add jekyll

  # Install gems
  bundle install


else
  echo "This script supports only UNIX, LINUX"
  echo "Exiting script"
  exit 1
fi
