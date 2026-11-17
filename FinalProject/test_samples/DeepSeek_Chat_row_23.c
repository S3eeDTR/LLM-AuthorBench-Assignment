#include <stdio.h>
#include <stdlib.h>

// Define the structure of a BST node
struct Node {
    int data;
    struct Node* left;
    struct Node* right;
};

// Function to create a new node
struct Node* createNode(int value) {
    struct Node* newNode = (struct Node*)malloc(sizeof(struct Node));
    newNode->data = value;
    newNode->left = NULL;
    newNode->right = NULL;
    return newNode;
}

// Function to insert a node into the BST
struct Node* insert(struct Node* root, int value) {
    if (root == NULL) {
        return createNode(value);
    }

    if (value < root->data) {
        root->left = insert(root->left, value);
    } else if (value > root->data) {
        root->right = insert(root->right, value);
    }

    return root;
}

// In-order traversal (Left, Root, Right)
void inorderTraversal(struct Node* root) {
    if (root != NULL) {
        inorderTraversal(root->left);
        printf("%d ", root->data);
        inorderTraversal(root->right);
    }
}

// Pre-order traversal (Root, Left, Right)
void preorderTraversal(struct Node* root) {
    if (root != NULL) {
        printf("%d ", root->data);
        preorderTraversal(root->left);
        preorderTraversal(root->right);
    }
}

// Post-order traversal (Left, Right, Root)
void postorderTraversal(struct Node* root) {
    if (root != NULL) {
        postorderTraversal(root->left);
        postorderTraversal(root->right);
        printf("%d ", root->data);
    }
}

// Main function
int main() {
    int arr[79] = {
        45, 23, 67, 12, 34, 56, 78, 9, 21, 33,
        44, 55, 66, 77, 88, 11, 22, 32, 43, 54,
        65, 76, 87, 98, 10, 20, 30, 40, 50, 60,
        70, 80, 90, 100, 13, 14, 15, 16, 17, 18,
        19, 24, 25, 26, 27, 28, 29, 31, 35, 36,
        37, 38, 39, 41, 42, 46, 47, 48, 49, 51,
        52, 53, 57, 58, 59, 61, 62, 63, 64, 68,
        69, 71, 72, 73, 74, 75, 79, 81, 82
    };

    struct Node* root = NULL;

    // Insert elements from the array into the BST
    for (int i = 0; i < 79; i++) {
        root = insert(root, arr[i]);
    }

    // Traverse and print the BST
    printf("In-order traversal: ");
    inorderTraversal(root);
    printf("\n");

    printf("Pre-order traversal: ");
    preorderTraversal(root);
    printf("\n");

    printf("Post-order traversal: ");
    postorderTraversal(root);
    printf("\n");

    return 0;
}