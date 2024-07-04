#include <stdio.h>
#include <string.h>

int main(){
    int i;
    scanf("%d", &i);
    char stars[i+1];
    strcpy(stars, "");
    for (int j = 1; j <= i; j++){
        strcat(stars, "*");
        printf("%s \n", stars);      
    } 
    printf("Hello World\n");
    return 0;
}