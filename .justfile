set shell := ["bash", "-c"]

# Data-regeneration recipes (offline augmentation pipelines). Dev workflow
# commands live in the Makefile.
import 'datagen.justfile'
