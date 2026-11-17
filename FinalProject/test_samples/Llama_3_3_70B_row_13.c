#include <stdio.h>

// Function to classify IP address
void classify_ip(char *ip) {
    int octet[4];

    // Extract octets from IP address
    sscanf(ip, "%d.%d.%d.%d", &octet[0], &octet[1], &octet[2], &octet[3]);

    // Check for valid IP address
    if (octet[0] < 0 || octet[0] > 255 || octet[1] < 0 || octet[1] > 255 || octet[2] < 0 || octet[2] > 255 || octet[3] < 0 || octet[3] > 255) {
        printf("Invalid IP address\n");
        return;
    }

    // Classify IP address
    if (octet[0] < 128) {
        printf("Class A IP address\n");
    } else if (octet[0] < 192) {
        printf("Class B IP address\n");
    } else if (octet[0] < 224) {
        printf("Class C IP address\n");
    } else if (octet[0] < 240) {
        printf("Class D IP address\n");
    } else {
        printf("Class E IP address\n");
    }
}

int main() {
    char ip[] = "192.168.90.160";
    printf("IP: %s\n", ip);
    classify_ip(ip);
    return 0;
}