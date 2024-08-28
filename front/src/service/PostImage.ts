
const uploadFile = async (
    apiUrl: string,
    endpoint: string,
    file: File,
    additionalData?: Record<string, any>
): Promise<Response> => {
    const formData = new FormData();
    formData.append('image', file);

    if (additionalData) {
        Object.keys(additionalData).forEach((key) => {
            formData.append(key, additionalData[key]);
        });
    }

    try {
        const response = await fetch(`${apiUrl}${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Error uploading file: ${response.statusText}`);
        }

        return response;
    } catch (error) {
        console.error('File upload failed:', error);
        throw error;
    }
};

export { uploadFile };
