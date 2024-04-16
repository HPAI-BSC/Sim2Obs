<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="images/space-logo.jpg" alt="Logo" width=100% height="300">
</div>

<h1 align="center">Space - CoE</h1>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#Launch">Launch</a>
      <ul>
        <li><a href="### MN5">Run mn5</a></li>
        <li><a href="### Local">Run local</a></li>
      </ul>
    </li>
  </ol>
</details>

## Launch

### MN5

```
LAUNCH_MN5_ACC <sbatch launch.sh> mn5 test attention U-Net
```

### Local

```
python src/train.py commit_tag=1 paths=local trainer=cpu
```