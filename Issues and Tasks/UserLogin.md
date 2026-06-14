# User Login

Have multiple users login at once. This doesn't have to be super complex, we only need about 5 at most. Additionally this is all on a local network, so nothing is going out onto the internet.

## Chroma DB

User HTTP Client instead of persistent client. It will still use the localhost IP, and use a different port #. This can be configured in the Toml files. Most everything else remains unchanged.

Connection to Chroma DB could be lost, but since it's all on one computer in a small network, it shouldn't be a problem.

## Gradio login

Gradio however, will change a bit.

Many of the states will be "user states." These are Session States, and should be close to what has already been written.

There only needs to be a few users.

Admin - can look at log tabs
User 1 - doesn't need to look at log tabs 
User 2 - doesn't need to look at log tabs

Since this is all meant to be local, this could be stored as a .toml file. That's a terrible idea, but since it's only for a few users and meant to separate them so that they can work concurrently, it's simple.