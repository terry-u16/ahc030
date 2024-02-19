use nalgebra::{DMatrix, DVector, DVectorView};

// 初回パラメータ調整分
const N1: &[u8] = b"AAAAAAAA8D/NzMzMzMzsPzMzMzMzM9M/zczMzMzM7D+amZmZmZnZPwAAAAAAAOA/mpmZmZmZyT9mZmZmZmbmPwAAAAAAAPA/zczMzMzM7D8AAAAAAADwPwAAAAAAAOA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/zczMzMzM7D/NzMzMzMzsP83MzMzMzOw/ZmZmZmZm5j+amZmZmZnJP5qZmZmZmdk/mpmZmZmZuT9mZmZmZmbmP2ZmZmZmZuY/AAAAAAAA4D9mZmZmZmbmP83MzMzMzOw/AAAAAAAAAAAAAAAAAAAAADMzMzMzM+M/mpmZmZmZyT8AAAAAAAAAAAAAAAAAAPA/AAAAAAAA8D/NzMzMzMzsP5qZmZmZmck/zczMzMzM7D+amZmZmZnZP83MzMzMzOw/zczMzMzM7D8AAAAAAADgPwAAAAAAAPA/mpmZmZmZyT8AAAAAAAAAAAAAAAAAAPA/MzMzMzMz0z+amZmZmZnJPwAAAAAAAAAAAAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/MzMzMzMz0z+amZmZmZnZP83MzMzMzOw/zczMzMzM7D9mZmZmZmbmPwAAAAAAAOA/MzMzMzMz4z9mZmZmZmbmPzMzMzMzM+M/MzMzMzMz0z8zMzMzMzPjP2ZmZmZmZuY/AAAAAAAA8D8zMzMzMzPTPzMzMzMzM+M/AAAAAAAA8D8zMzMzMzPjP5qZmZmZmck/mpmZmZmZyT8AAAAAAADgPzMzMzMzM+M/mpmZmZmZuT8zMzMzMzPjPwAAAAAAAPA/zczMzMzM7D+amZmZmZm5PzMzMzMzM+M/mpmZmZmZyT+amZmZmZnZP5qZmZmZmbk/MzMzMzMz0z+amZmZmZnZPzMzMzMzM+M/zczMzMzM7D8zMzMzMzPTPwAAAAAAAAAAzczMzMzM7D+amZmZmZm5P5qZmZmZmbk/";
const M1: &[u8] = b"zTt/Zp6g1j/q+NKpfyrlPwAAAAAAAOA/6vjSqX8q5T/NO39mnqDmP6j0l5t34+E/zTt/Zp6g1j+qTFjoerbrP6pMWOh6tus/AAAAAAAA4D/aTk+x3vvuP1Nb2jpYTOk/2k5Psd777j+o9Jebd+PhPwAAAAAAAOg/zTt/Zp6g5j/ZbN/MdvjwP807f2aeoNY/qPSXm3fj4T8uIQkUjpjjPy4hCRSOmOM/zTt/Zp6g1j/zWWFEK9jsP0k/aBHq7u0/LiEJFI6Y4z/q+NKpfyrlP0k/aBHq7u0/zTt/Zp6g1j+qTFjoerbbP1Nb2jpYTOk/qkxY6Hq22z+qTFjoerbbP9pOT7He++4/81lhRCvY7D+qTFjoerbrP6pMWOh6tts/2WzfzHb48D8AAAAAAADoP807f2aeoOY/zTt/Zp6g5j+o9Jebd+PhP1Nb2jpYTOk/6vjSqX8q5T+qTFjoerbbP9pOT7He++4/6vjSqX8q5T/q+NKpfyrlPwAAAAAAAOA/zTt/Zp6g5j8H7a9mD37wP/lNZDSDb/E/zTt/Zp6g5j8AAAAAAADoP6j0l5t34+E/AAAAAAAA6D80b/2TcojqPy4hCRSOmOM/qPSXm3fj4T+o9Jebd+PhP807f2aeoNY/qPSXm3fj4T+qTFjoerbrP+r40ql/KuU/zTt/Zp6g5j8uIQkUjpjjP6pMWOh6tus/AAAAAAAA6D/NO39mnqDWP+r40ql/KuU/zTt/Zp6g1j+o9Jebd+PhP807f2aeoOY/qkxY6Hq22z8AAAAAAADoPwftr2YPfvA/zTt/Zp6g1j+qTFjoerbbP6j0l5t34+E/AAAAAAAA4D8AAAAAAADoPwAAAAAAAOA/zTt/Zp6g1j8uIQkUjpjjP6j0l5t34+E/NG/9k3KI6j+o9Jebd+PhPwAAAAAAAOA/2k5Psd777j8uIQkUjpjjPwAAAAAAAOA/";
const EPS1: &[u8] = b"mpmZmZmZuT9nZmZmZmbmP2ZmZmZmZu4/mpmZmZmZ6T9nZmZmZmbWPzMzMzMzM9M/mpmZmZmZ4T8AAAAAAADwP5qZmZmZmeE/ZmZmZmZm7j9nZmZmZmbmP5qZmZmZmek/mpmZmZmZ4T8zMzMzMzPDPzMzMzMzM9M/mpmZmZmZyT9nZmZmZmbWP2dmZmZmZtY/zczMzMzM5D/MzMzMzMzsP5qZmZmZmeE/AAAAAAAA6D8AAAAAAADoP5qZmZmZmbk/Z2ZmZmZm1j8zMzMzMzPDP2ZmZmZmZu4/zMzMzMzM3D8AAAAAAADwPzQzMzMzM+s/mpmZmZmZ2T+amZmZmZnZP8zMzMzMzOw/mpmZmZmZ2T/NzMzMzMzkP5qZmZmZmeE/MzMzMzMzwz+amZmZmZm5PzQzMzMzM+s/mpmZmZmZuT8zMzMzMzPjPwAAAAAAANA/zczMzMzM5D/NzMzMzMzkP2ZmZmZmZu4/AAAAAAAA0D+amZmZmZnZPzMzMzMzM9M/zczMzMzM5D+amZmZmZmpP2ZmZmZmZu4/NDMzMzMz6z8AAAAAAADQPwAAAAAAAPA/NDMzMzMz6z+amZmZmZnZP5qZmZmZmek/MzMzMzMz0z/NzMzMzMzkP5qZmZmZmak/mpmZmZmZ4T8zMzMzMzPjP5qZmZmZmeE/MzMzMzMz4z8zMzMzMzPTP5qZmZmZmeE/MzMzMzMzwz+amZmZmZnZPzMzMzMzM8M/AAAAAAAA0D8AAAAAAADwP2dmZmZmZtY/AAAAAAAA6D+amZmZmZnpP5qZmZmZmck/mpmZmZmZ6T9nZmZmZmbWP5qZmZmZmak/AAAAAAAA4D80MzMzMzPrP2dmZmZmZtY/mpmZmZmZuT+amZmZmZnhP2dmZmZmZtY/mpmZmZmZ2T9nZmZmZmbmPzMzMzMzM9M/MzMzMzMz4z8zMzMzMzPTP83MzMzMzOQ/";
const AVG1: &[u8] = b"ST9oEeru7T8AAAAAAADgP6j0l5t34+E/NG/9k3KI2j80b/2TcojaP9pOT7He+94/hR4VuY5U4j8uIQkUjpjTP/NZYUQr2Nw/NG/9k3KI6j8uIQkUjpjTP+r40ql/KtU/AAAAAAAA2D+io7MN5VLnP9ls38x2+OA/+U1kNINv4T/NO39mnqDWP191vH6/v+8/FgR3de4u4z80b/2TcojaP1Nb2jpYTNk/ST9oEeru3T/q+NKpfyrVPwAAAAAAANg/zTt/Zp6g1j9TW9o6WEzZP1Nb2jpYTNk/AAAAAAAA5D8AAAAAAADYPy4hCRSOmNM/2WzfzHb44D/q+NKpfyrVP6j0l5t349E/qkxY6Hq22z8uIQkUjpjTP4UeFbmOVOI/qPSXm3fj0T8uIQkUjpjTPwAAAAAAAOA/qPSXm3fj4T9JP2gR6u7dP1Nb2jpYTNk/6vjSqX8q1T8uIQkUjpjTP6j0l5t349E/qkxY6Hq22z+o9Jebd+PRP6pMWOh6tts/NG/9k3KI2j80b/2TcojaP6j0l5t349E/AAAAAAAA0D8uIQkUjpjTP0k/aBHq7t0/AAAAAAAA4D80b/2TcojaP+r40ql/KtU/AAAAAAAA5D80b/2TcojaP+r40ql/KuU/81lhRCvY3D8AAAAAAADYPwftr2YPfuA/ST9oEeru3T/NO39mnqDWPy4hCRSOmNM/zTt/Zp6g1j/zWWFEK9jsPy4hCRSOmNM/B+2vZg9+4D8AAAAAAADYPy4hCRSOmNM/ST9oEeru3T/NO39mnqDWP807f2aeoNY/ZBImSkfs6T80b/2TcojaP9pOT7He+94/2WzfzHb44D8uIQkUjpjTPy4hCRSOmNM/j6U20q3o5T9TW9o6WEzZPxYEd3XuLuM/6vjSqX8q1T80b/2TcojaPzRv/ZNyiNo/NG/9k3KI2j/NO39mnqDWP6pMWOh6tts/";
const K: &[u8] = b"g5yDStk8HECfhjvSvhsMQMBHXWZUkvQ/pTrbWwPTAUBQ5pg224ICQB2T+CgXIwdA18yrtEvUBEAsIRd0C8YCQKpMWOh6tvs/1M3T8B2GEkBR5hgKgr0GQKPHujwKQvs/8IA/SKsFBUDRa3eoZUUWQPOXY1hifARAfEDnyoP5CUB1n9dmXGf5P5SODYVrORxAysmaOPJoEUCqTFjoerb7Pw8hXDUJh/Y/tMkT94OQ/z/33PKICJvpP6Yv6kszYf4/qkxY6Hq2+z/4V/FIuOMQQODD+oaTnwJAqvWVfdCpG0AoezlcPDAFQOzlkVQL8+s/oF3wv0CXGUC5DpjmMvMTQDfe//aNaAVAv1CxEBOc+T/NCkT1t//tPzrsEyb9hRpAwQrhchI/AUBUx4EfDDcGQMwynlUNcwBAunT/EpSzD0CZ7lioAfoMQFOi5thGlQBA9g0UJqGV6T98sL6rv5EEQGSoo0mdlAJAKwS7IIX9DUCdmJvHAmjzP+gx6wJkkhNA1gSNHWGq/z9++0dcAP4DQDgPzaKDafo/zTt/Zp6g5j+lXw2WWfvwPyLJURiRagNAu763Avvy5z9YkBn6CqP7P/Ox6GPaYvo/0DxMdeVlE0AERXME1wACQCyBqpU5/wNAlBdKi8ChFECh1pFKKx7nP0118AFaZQdAZru3Fl8W+z+hjm9jz5T8P84rrMHLVuw/t28XnDEtBECAjpL15S0TQA/HS5UuiwdANh3TkD+n+z+gwPszarYHQMs1mBzl0wFAdRzuuZmP9T8q/Ba9njP3PyselC5UcQBA4Xgwpu/9FkDQ3lY6uej/PwWsMA53sBdA7ehEO76u+j8W8ra3Hvz0P38CRu8BYwdAhJzco835FkBo+pM6gO4CQEViIAVIYwRAQD1wOpqD+j/I/uZmPhUEQGnPrH7IFwNAs0oHfZsqBEDylVEAM1cGQKK8MenQa+4/";
const B: &[u8] = b"MG8+3AupoT8AXYjUR6OQPywgAszZVqc/fKQlTKxOuD+Wammbt52pPzVHf6uAjJw/Y1L+qrKtqj8SAe4zUmirP5qZmZmZmbk/Aq6bty4Brj91RPbPC5+gPwb28E028rQ/sMYzhGaNlD+AKNZSde8mP8QqK/J7Pbk/AL5osqonoz+6/4JTH1GzP0Ozk7YdYqo/SLe1uKEXsz+amZmZmZm5P8ip42YwZYc/pnpNeV5gpD/qtwgYt7SdP71COgzIv50/mpmZmZmZqT/Y/ywQHgGuPw9sj/i1yrM/fVU8Vzh5mz9TYAp3E3SiP39jcJSj8KY/kA/MN6aqsD8lbUaFZnmkP6CyVxLjmHQ/p5efinq7nT9wQx3KNL+GPw39hztsfK4/nL9mwPHUoD8o+lYm5C2PPzI5YbqUKLM/TAsWI+4ZkT8AWoWw38ERP5aWVn19maY/5wlqx18Zmz+Sy9aqmneyPyQeNu6m96U/PKmquj4Poj+9MOJLKECXP5WOAboGCqI/nPWvZAL1oz/Z9Wyf6N2iP7tyy1nfFJg/0Mb30bULqj+s1sE3/U6hP4AM3Uh5hKc/wIDNDK6Mjz8z90Jr2H+vP8VM/zBh5KQ/HbTUoTVLpT/ZbaqJGY6nP//KB1Orf6A/XRtkCgk/kj/Wj3+XrGW4P0sxuxcyP6o/L9U0D7D2nj/Fg57CFIK1P8pD0jGVLqg/dNfPTC4yoz8Jo8UMqPGwP7DAgNAJHak/uwCCTMG4iD+6wIHam8KQP+j3RYbzKWA/hEml83Lunj+hQaFgrTSpPwa6PqbVtqY/CCHyeOZVpT/ZD2YImRSyP/wrshY/FqA/PzK/LWMxqD/goW4UKI2pPwwK3pTFI5k/qOw7OzgvpD9yZpCtKMilP1pWefpPZKc/xi41xM6OsD/BZikAgaymP19Xa+VMyKQ/iH9gi2b7jD8/b6lLLPagP/h7QMygx60/";
const R: &[u8] = b"zV1nPHWPyT959QoO8S7jP/TUeApNaOA/+BgVBbo91z+cho15kSnqPzzkZCfZ1OM/NJSFT98Q2j8osDUZMjDRP2ZmZmZmZuY/65v1qTgLwj8J46UPpgPZPyA9J6rm/uc/FsqtSauJ6j95jHfCGRLkP/cfIFdxXeQ/zM7Cuxjw6j+TUHxIbS3QPyy2ow32jd4/4vp1zdd91T9mZmZmZmbmPyQz9vhHYuM/0XCsJjP/5D8zeonLeCTpP3+A0LiTnuw/ZmZmZmZm5j/uXl8mTyPiP98pKTIOqOU/rM2pRusl5z8wqM4t2mXoP+hViO5yG+U/WEPdMd4t5z/XBOoztk3oPxwCD+T6UdY/o8lpfS4m5z8y4ATi8/7lP16QvgojM9k/dWfuM5JX1D+KeaLZcMzoPwKN1RVthd0/hl5qVhUf6D8cghr+xNnSPwYnSYkgP+U/mpZoANmn5z/tY+q0LnniPxKMiLTnodk/XaOkh4PA6T9W3buC3t3iPwJYsi7weuc/d8oxASyE5j81qx23QNHfP8X9911ec9g/tgQCHtF55D86QWmZhefrP+BwNYGPnc4/NwOA2uP94D9aitSJGQ7oPxYmAXCSAd8/ZKvu4gdZ4z82rDDMtL/bPy5jVAuxpdI/6FVu4ayj1z9/YgTatrrpPxUXe6oQJeM/0ig4e08C5z83PtJ9HY7mP15WY9MKYek/ywqpwQMg4z/agwwWcO3pP4Lvne9QX+k/z8qK1leZzT/h347t3uzkP0BJ0hixO+c/h4nnlIIuvD+wj7tE2K3jP0w9weogZNs/57UNp1920z+IIF1BqtPjP9TbSA/RVuY/JZTeM71k5D+ntdrxwWblPwzqpdM9fLo/EDsaHK3O2T8tvSqNB9fhP560bMXEh+E/yZAWVxpG5T/9KfA8XL7fP7pR2fWa/us/EqYe95Vruj+7ePuU+H3nPxCyu7oZ978/";
const MULTI: &[u8] = b"AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAAAAAAAAAAAAAPA/AAAAAAAA8D8AAAAAAAAAAAAAAAAAAPA/AAAAAAAAAAAAAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAAAAAAAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAAAAAAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAAAAAAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAAAAAAAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAAAAAAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAAAAAAAAAAAAAAAAAAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAAAAAAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAA8D8AAAAAAADwPwAAAAAAAPA/AAAAAAAAAAAAAAAAAADwPwAAAAAAAPA/";
const PARAM_K: &[u8] = b"2qbZ71+CF0ComgG6gfbVP7Q/WeAhY+I/FtpsUZTW8z8=";
const PARAM_B: &[u8] = b"ra5HgjkQhz8et+IitFqHPy8JYNrO9u8/3n8aUaKChD8=";
const PARAM_R: &[u8] = b"nEbw0j98hD9iwKbI6S6WP8xsXBtvwsM/OPLKPXR1mT8=";
const PARAM_MULTI: &[u8] = b"orrL5ghhsD8o6BdPpba4P9czAbXu07o/6xfyq5SKhD8=";

// 第2回パラメータ調整分
const N2: &[u8] = b"ZmZmZmZm5j9mZmZmZmbmP5qZmZmZmdk/ZmZmZmZm5j+amZmZmZm5P5qZmZmZmek/AAAAAAAAAACamZmZmZm5PzMzMzMzM+M/MzMzMzMz0z+amZmZmZnpPwAAAAAAAAAAmpmZmZmZ2T/NzMzMzMzsP5qZmZmZmdk/mpmZmZmZyT+amZmZmZnJP5qZmZmZmdk/zczMzMzM7D8AAAAAAADgPwAAAAAAAOA/AAAAAAAA4D8zMzMzMzPjP2ZmZmZmZuY/AAAAAAAAAABmZmZmZmbmPwAAAAAAAOA/AAAAAAAAAACamZmZmZnpPwAAAAAAAAAAmpmZmZmZuT8AAAAAAADgP5qZmZmZmek/MzMzMzMz4z+amZmZmZnJPwAAAAAAAAAAZmZmZmZm5j8zMzMzMzPjPwAAAAAAAOA/MzMzMzMz4z8AAAAAAAAAAJqZmZmZmdk/AAAAAAAAAAAAAAAAAADgPwAAAAAAAAAAmpmZmZmZyT+amZmZmZm5P5qZmZmZmck/mpmZmZmZ2T8zMzMzMzPjP5qZmZmZmek/mpmZmZmZyT+amZmZmZnJP5qZmZmZmek/mpmZmZmZ6T8zMzMzMzPTPwAAAAAAAAAAmpmZmZmZuT+amZmZmZnJP5qZmZmZmbk/mpmZmZmZ6T8zMzMzMzPjPwAAAAAAAAAA";
const M2: &[u8] = b"LiEJFI6Y4z8AAAAAAADoP807f2aeoOY/AAAAAAAA4D8uIQkUjpjjPzRv/ZNyiOo/zTt/Zp6g1j/NO39mnqDWP6pMWOh6tus/6vjSqX8q5T8uIQkUjpjjP6pMWOh6tts/LiEJFI6Y4z+qTFjoerbrPwAAAAAAAOA/LiEJFI6Y4z+qTFjoerbbP807f2aeoOY/NG/9k3KI6j8uIQkUjpjjPy4hCRSOmOM/qkxY6Hq22z/NO39mnqDmP+r40ql/KuU/zTt/Zp6g1j+o9Jebd+PhP1Nb2jpYTOk/zTt/Zp6g1j+o9Jebd+PhPwAAAAAAAOA/zTt/Zp6g1j+qTFjoerbbP1Nb2jpYTOk/NG/9k3KI6j/q+NKpfyrlP6j0l5t34+E/zTt/Zp6g5j9TW9o6WEzpP6j0l5t34+E/LiEJFI6Y4z+o9Jebd+PhPwAAAAAAAOA/qkxY6Hq22z80b/2TcojqP6pMWOh6tts/LiEJFI6Y4z+qTFjoerbbPwAAAAAAAOA/LiEJFI6Y4z+o9Jebd+PhP807f2aeoOY/qPSXm3fj4T+o9Jebd+PhP+r40ql/KuU/AAAAAAAA4D+o9Jebd+PhP807f2aeoNY/LiEJFI6Y4z8uIQkUjpjjP6j0l5t34+E/zTt/Zp6g1j8AAAAAAADoP807f2aeoNY/";
const EPS2: &[u8] = b"Z2ZmZmZm1j+amZmZmZnpP5qZmZmZmak/MzMzMzMz4z/MzMzMzMzcPwAAAAAAAOg/zMzMzMzM3D9mZmZmZmbuP2dmZmZmZuY/AAAAAAAA0D8AAAAAAADoP8zMzMzMzOw/mpmZmZmZuT/NzMzMzMzkP2dmZmZmZuY/MzMzMzMz0z9nZmZmZmbmP5qZmZmZmbk/AAAAAAAA6D8zMzMzMzPDPzMzMzMzM8M/Z2ZmZmZm1j9nZmZmZmbWPzQzMzMzM+s/Z2ZmZmZm1j8AAAAAAADQP2ZmZmZmZu4/MzMzMzMz4z80MzMzMzPrPwAAAAAAAOg/MzMzMzMz4z9mZmZmZmbuPzMzMzMzM+M/zMzMzMzM7D8AAAAAAADQP83MzMzMzOQ/AAAAAAAA4D8AAAAAAADoPzMzMzMzM9M/mpmZmZmZ2T+amZmZmZnhPzMzMzMzM9M/MzMzMzMzwz80MzMzMzPrPwAAAAAAAOg/Z2ZmZmZm1j/MzMzMzMzsPwAAAAAAAOA/mpmZmZmZ6T/NzMzMzMzkPzMzMzMzM9M/ZmZmZmZm7j8AAAAAAADQPzMzMzMzM9M/Z2ZmZmZm1j9nZmZmZmbWP5qZmZmZmek/zMzMzMzM3D8zMzMzMzPTP2ZmZmZmZu4/AAAAAAAA8D9mZmZmZmbuP2dmZmZmZuY/";
const AVG2: &[u8] = b"AAAAAAAA2D8AAAAAAADgPwAAAAAAANA/w8FoBWRF5j8AAAAAAADYPy4hCRSOmNM/p1ZUWfzC4j+o9Jebd+PhP1Nb2jpYTNk/AAAAAAAA0D8WBHd17i7jP0k/aBHq7t0/AAAAAAAA4D80b/2TcojaPy4hCRSOmOM/U1vaOlhM2T9JP2gR6u7dP6pMWOh6tts/6vjSqX8q1T+qTFjoerbbP9ls38x2+OA/2k5Psd773j/zWWFEK9jcP/NZYUQr2Nw/ST9oEeru3T/2LxJfZWXkPwAAAAAAANA/AAAAAAAA5D8AAAAAAADkPy4hCRSOmNM/AAAAAAAA5D+AOUIu3MjkPwAAAAAAANg/qPSXm3fj0T8AAAAAAADYPwAAAAAAANg/B+2vZg9+4D8AAAAAAADYPzRv/ZNyiNo/2k5Psd773j8uIQkUjpjTP6j0l5t34+E/AAAAAAAA2D/NO39mnqDWPwAAAAAAAOA/AAAAAAAA2D/aTk+x3vveP+r40ql/KtU/2k5Psd773j8AAAAAAADgP/NZYUQr2Nw/NG/9k3KI2j9JP2gR6u7dPzRv/ZNyiNo/B+2vZg9+4D80b/2TcojaPwftr2YPfuA/U1vaOlhM2T/q+NKpfyrVP+r40ql/KtU/Ek+N6/t+7z9TW9o6WEzZPwAAAAAAAOQ/";
const TABOO: &[u8] = b"+1gXvrDi9T/Gmio5OFj3v5Kikez7F+m/0rVLsFFf6r+tszH50rIKwEuTCl+6wQXARkDCIM9C7L8pgpsp2sD2v7vwSvcIreE/AxALcBX+0L/SEFrCj9v0v0zcdpXzd/S/AAAAAAAAAAAu/Lyyvo3Tv74y5JAt7Oc/AAAAAAAAAAAkYz5YqnP7Pw8UeQ9nUwTAdFJWnCYVEMAR25sKRt/hv9gTQBdhvghAGQvJ2EqJCUCr1wDD57oDwJHDLiBOuQJA5v1pnG6T8D8RD+rrLBoPQHUBhmyfk/K/UYYwb+lz9L8yGZgz5Db6P4ntE21Whuy/0x9DzC068D884hW2ZPvoP/67kvEO9RbANyiV1VyT1T+kcKIoi1YXwFI2al86mADAnwi3TDWF5j+C3CjckQsKQNV4di9Ot/E//DXvW/YO+b8OtpzV/f7xv7zsfwHRFuo/hlEepHtZ/T8V5DiuTMrwv5mv56tmb/2/NV8AtJt38z9fV6AQacL0PxheKZOiRhBAfaJBOxTuFsBew0nLpysKQKJLQbjgTrA/IUFfcI1xBMB0lVkw+6DpvxdulTKP1RlAo0A25H3X6b8Zvj6n+n//P1F/o8rSBQvANrRg2Kyz2L8gbHDiopUAwGlyu9/ovAbAzg4/jWXnlT8LC8HymK7zPyjp6geEi+6/";
const ENTROPY_LEN: &[u8] = b"/O5jaTPVKkC0uD1pQegqQFqdKgCqWiVAL6TjXG27LUAjuXGCEsYzQB5f50zdVC5AzsIVU8KwKUCIgHbVXJckQNpOT7He+x5AU1vaOlhMKUBbefdG90MuQOARPG3OaCdAwAofAMZIHEBBh2LGMt4iQIUeFbmOVCJAwAofAMZIHEBM+chUwNouQPYvEl9lZTRAQLAaxKm/J0CnVlRZ/MIiQDRv/ZNyiBpAxUJsP1eHLkCAOUIu3MgUQDf5z9pv4SRAp1ZUWfzCIkDQumE/4B8gQMbCdq/ucTRAI9BwHBshG0DkEORzVw8pQAAAAAAAACxAxrENHtVkHUBAVmNFSi0vQCkD7C9DEixAqPSXm3fjEUBiKcDzoz82QKj0l5t34zFA+U1kNINvMUCqxLqmTds0QBwkAtahvzVAzsIVU8KwKUBwxJ05jXYuQMbCdq/ucTRAWCgvhHmuMkAE8W2P3KkxQFZU22QuUDNAV7LbGpqlM0ADq/e2g8U1QGJWY32dsjNA7S4oiiSRNEDGwnav7nE0QB5f50zdVC5ASerUGpJGMkBm7kCfAx4xQNqpUl2+uTVAwlzxgehCM0AI9MvQoMwtQHcvDo0vcyZAp1ZUWfzCMkAG+5VNMNowQGChHSrROipALgAy9EH7KkAtwpPREKoXQPqB/oV6JCxA";
const PARAM_TABOO: &[u8] = b"1CzxLdPSkz/dEoBpRzn2P6bJtg6xvZU/j/BK4GuXEEA=";
const PARAM_ENTROPY_LEN: &[u8] = b"MYOjtt/qhz+qdYAoiFCaP/0Ucx+5iKQ/dURErWFFN0A=";

pub struct ParamSuggester {
    x_matrix: DMatrix<f64>,
    y_vector: DVector<f64>,
    hyper_param: DVector<f64>,
    y_inv_trans: fn(f64) -> f64,
    lower: f64,
    upper: f64,
}

impl ParamSuggester {
    fn new(
        hyper_param: DVector<f64>,
        x_matrix: DMatrix<f64>,
        y_vector: DVector<f64>,
        y_inv_trans: fn(f64) -> f64,
        lower: f64,
        upper: f64,
    ) -> Self {
        Self {
            hyper_param,
            x_matrix,
            y_vector,
            y_inv_trans,
            lower,
            upper,
        }
    }

    fn gen_x_matrix_1() -> DMatrix<f64> {
        let n = DVector::from_vec(decode_base64(N1)).transpose();
        let m = DVector::from_vec(decode_base64(M1)).transpose();
        let eps = DVector::from_vec(decode_base64(EPS1)).transpose();
        let avg = DVector::from_vec(decode_base64(AVG1)).transpose();

        let x_matrix = DMatrix::from_rows(&[n, m, eps, avg]);

        x_matrix
    }

    fn gen_x_matrix_2() -> DMatrix<f64> {
        let n = DVector::from_vec(decode_base64(N2)).transpose();
        let m = DVector::from_vec(decode_base64(M2)).transpose();
        let eps = DVector::from_vec(decode_base64(EPS2)).transpose();
        let avg = DVector::from_vec(decode_base64(AVG2)).transpose();

        let x_matrix = DMatrix::from_rows(&[n, m, eps, avg]);

        x_matrix
    }

    pub fn gen_k_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_K));
        let y_vector = DVector::from_vec(decode_base64(K));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| x * x,
            0.0,
            50.0,
        )
    }

    pub fn gen_b_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_B));
        let y_vector = DVector::from_vec(decode_base64(B));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| x,
            0.0,
            1.0,
        )
    }

    pub fn gen_r_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_R));
        let y_vector = DVector::from_vec(decode_base64(R));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| x,
            0.1,
            0.9,
        )
    }

    pub fn gen_multi_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_MULTI));
        let y_vector = DVector::from_vec(decode_base64(MULTI));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_1(),
            y_vector,
            |x| x,
            0.0,
            1.0,
        )
    }

    pub fn gen_taboo_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_TABOO));
        let y_vector = DVector::from_vec(decode_base64(TABOO));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_2(),
            y_vector,
            |x| x.exp() / (1.0 + x.exp()),
            0.0,
            1.0,
        )
    }

    pub fn gen_entropy_len_pred() -> Self {
        let hyper_param = DVector::from_vec(decode_base64(PARAM_ENTROPY_LEN));
        let y_vector = DVector::from_vec(decode_base64(ENTROPY_LEN));
        Self::new(
            hyper_param,
            Self::gen_x_matrix_2(),
            y_vector,
            |x| x * x,
            20.0,
            500.0,
        )
    }

    pub fn suggest(&self, n: usize, m: usize, eps: f64, avg: f64) -> f64 {
        let n = (n - 10) as f64 / 10.0;
        let m = (m as f64).sqrt() / 4.0;
        let eps = eps * 5.0;
        let avg = avg.sqrt() / 8.0;

        let len = self.x_matrix.shape().1;
        let y_mean = self.y_vector.mean();
        let y_mean = DVector::from_element(self.y_vector.len(), y_mean);
        let new_x = DMatrix::from_vec(4, 1, vec![n, m, eps, avg]);
        let noise = DMatrix::from_diagonal_element(len, len, self.hyper_param[3]);

        let k = self.calc_kernel_matrix(&self.x_matrix, &self.x_matrix) + noise;
        let kk = self.calc_kernel_matrix(&self.x_matrix, &new_x);

        let kernel_lu = k.lu();
        let new_y = kk.transpose() * kernel_lu.solve(&(&self.y_vector - &y_mean)).unwrap();

        (self.y_inv_trans)(new_y[(0, 0)] + y_mean[(0, 0)]).clamp(self.lower, self.upper)
    }

    fn calc_kernel_matrix(&self, x1: &DMatrix<f64>, x2: &DMatrix<f64>) -> DMatrix<f64> {
        let n = x1.shape().1;
        let m = x2.shape().1;
        let mut kernel = DMatrix::zeros(n, m);

        for i in 0..n {
            for j in 0..m {
                kernel[(i, j)] = self.gaussian_kernel(&x1.column(i), &x2.column(j));
            }
        }

        kernel
    }

    fn gaussian_kernel(&self, x1: &DVectorView<f64>, x2: &DVectorView<f64>) -> f64 {
        let t1 = self.hyper_param[0];
        let t2 = self.hyper_param[1];
        let t3 = self.hyper_param[2];

        let diff = x1 - x2;
        let norm_diff = diff.dot(&diff);
        let dot = x1.dot(&x2);
        t1 * dot + t2 * (-norm_diff / t3).exp()
    }
}

fn decode_base64(data: &[u8]) -> Vec<f64> {
    const BASE64_MAP: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut stream = vec![];

    let mut cursor = 0;

    while cursor + 4 <= data.len() {
        let mut buffer = 0u32;

        for i in 0..4 {
            let c = data[cursor + i];
            let shift = 6 * (3 - i);

            for (i, &d) in BASE64_MAP.iter().enumerate() {
                if c == d {
                    buffer |= (i as u32) << shift;
                }
            }
        }

        for i in 0..3 {
            let shift = 8 * (2 - i);
            let value = (buffer >> shift) as u8;
            stream.push(value);
        }

        cursor += 4;
    }

    let mut result = vec![];
    cursor = 0;

    while cursor + 8 <= stream.len() {
        let p = stream.as_ptr() as *const f64;
        let x = unsafe { *p.offset(cursor as isize / 8) };
        result.push(x);
        cursor += 8;
    }

    result
}
