# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"

  config.vm.provider "virtualbox" do |vb|
    # the more RAM the better
    vb.memory = "4096"
  end
  
  config.vm.provision "shell", inline: <<-SCRIPT
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y python3-pip
  SCRIPT

  config.vm.provision "shell", privileged: false, inline: <<-SCRIPT
  python3 -m pip install gavel gensim
  SCRIPT
end