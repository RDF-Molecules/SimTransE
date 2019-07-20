# SimTransE
We present SimTransE approach that exploits both similarities between entities, e.g., drugs and target, as well as their connections in a knowledge graph.
From interactions and similarities between entities, SimTransE embeds entities into the high-dimensional vector space.
SimTransE optimizes the distance between these vector embeddings, considering the existing interactions between drugs and targets, drug-drug, and target-target similarities to then predict unknown interactions. We conduct evaluations to assess the quality of SimTransE with respect to TransE approach that just considers interactions among entities. 
Observed results suggest that SimTransE is able to outperform TransE, taking advantage of the [homophily theory](https://en.wikipedia.org/wiki/Homophily). 

## Architecture
![architecture](https://user-images.githubusercontent.com/4923203/61579241-4dcc5880-ab03-11e9-9249-11a035670078.png)

## Learning the embeddings

### Algorithm 1

![algorithm_1](https://user-images.githubusercontent.com/4923203/61579337-14481d00-ab04-11e9-8233-8238064fab35.png)

### Functions

![functions](https://user-images.githubusercontent.com/4923203/61579344-2de96480-ab04-11e9-8111-a796c96190c9.png)

### Algorithm 2

![algorithm_2](https://user-images.githubusercontent.com/4923203/61579348-4194cb00-ab04-11e9-8ffb-dc5a74c76dd2.png)

## Evaluation

### Formulas

![formulas](https://user-images.githubusercontent.com/4923203/61579406-d4356a00-ab04-11e9-8718-1cd34298ef2a.png)
