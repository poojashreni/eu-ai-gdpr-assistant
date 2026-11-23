# LEARNING.md — How to read this project

This file is your guide to what we build and why. The project is a chat assistant that answers questions about the EU AI Act and the GDPR, in German or English, and always points to the exact article it used.

We build it in eight stages. Each stage adds one idea and (from stage 1 on) comes with a notebook you can run, and a section below that explains it in plain words.

How to use it:

1. Read the section for the stage first, then open the matching notebook.
2. Run the notebook cell by cell. Change something and see what happens: that is the fastest way to learn.
3. Do not move on until the stage makes sense. Each later stage builds on the earlier ones.

The technique behind it is called RAG (retrieval-augmented generation): instead of letting a language model answer from memory, we first look up the relevant passages of the law, then ask the model to answer using only those passages. That is how the answers stay grounded and can be cited.

## §1 Load & parse

## §2 Chunking comparison

## §3 Embeddings + Chroma

## §4 Hybrid retrieval

## §5 RAG chain

## §6 Evaluation

## §7 API + UI

## §8 Docker, CI, deployment
