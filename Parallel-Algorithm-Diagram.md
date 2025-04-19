graph TD
    A[Matrix A size m x n] --> |Sequential| D[Compute matrix product using 3-level loop]
    B[Matrix B size n x p] --> |Sequential| D
    D --> E[Result matrix C size m x p]

    A --> |Shared Memory| F[Split by rows]
    B --> |Shared Memory| G[Broadcast entire matrix B]
    F --> H1[Process 1: Compute rows 1..k]
    F --> H2[Process 2: Compute rows k+1..m]
    G --> H1
    G --> H2
    H1 --> I[Merge results from processes]
    H2 --> I
    I --> J[Result matrix C size m x p]

    A --> |Distributed Memory| K[Scatter by rows]
    B --> |Distributed Memory| L[Broadcast entire matrix B]
    K --> M1[MPI Process 1: Compute rows 1..j]
    K --> M2[MPI Process 2: Compute rows j+1..m]
    L --> M1
    L --> M2
    M1 --> N[Gather results using MPI]
    M2 --> N
    N --> O[Result matrix C size m x p]