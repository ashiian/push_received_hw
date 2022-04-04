build: 
	docker build -t best_push_time:dev .

test: 
	docker run best_push_time:dev "python3.8" /srv/src/hello.py

run: 
	docker run -v /Users/ashiyan/Documents/funcorp/data/:/srv/src/data/ best_push_time:dev "python3.8" /srv/src/train.py 

start: build run
