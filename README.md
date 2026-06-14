# vrfcd

`vrfcd` stands for **van Rossum Functional Community Detection**.

It provides a pipeline for converting spike train data into functional networks using the van Rossum distance and then detecting functional communities.

## Workflow

```text
spike trains -> van Rossum distance -> functional matrix -> graph -> communities
