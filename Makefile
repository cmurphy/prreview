.PHONY: all build

all: build

build:
	go build -o prreview main.go

install:
	go install
